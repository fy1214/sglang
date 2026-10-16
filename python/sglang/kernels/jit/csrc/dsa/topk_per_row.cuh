// SPDX-License-Identifier: Apache-2.0
// Port of vLLM topKPerRow (csrc/libtorch_stable/sampler.cu) for SGLang kpool select.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cub/cub.cuh>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace sglang {
namespace topk_per_row {

template <int step>
static inline __device__ uint32_t extractBinIdx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 5;
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

    if constexpr (step == 1) {
      return bits >> 21;
    } else if constexpr (step == 2) {
      return (bits >> 10) & 0x7ff;
    } else if constexpr (step == 3) {
      return bits & 0x3ff;
    }
  }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

/**
 * Map a Func over the input data, using vectorized load instructions if
 * possible.
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param thread_rank rank of the calling thread among all participating threads
 * @param num_threads number of the threads that participate in processing
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, const T* in, idxT len, Func f) {
  // Use dynamic WARP_SIZE from cuda_compat.h to support both
  // Wave64 (MI300X/gfx942) and Wave32 (Strix Halo/gfx1151) architectures
  constexpr int kWarpSize = WARP_SIZE;
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    // TODO: it's UB
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
                       ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
                       : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(kWarpSize >= items_per_scalar);
    // and because items_per_scalar > skip_cnt, kWarpSize > skip_cnt
    // no need to use loop
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    // because len_cast = (len - skip_cnt) / items_per_scalar,
    // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
    // and so
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
    // kWarpSize no need to use loop
    const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

template <
    int step,
    int kNumThreadsPerBlock,
    int kNumBins,
    int kNumFinalItems,
    bool multipleBlocksPerRow,
    bool mergeBlocks,
    bool useTopK512Optimization,
    typename SmemFinalType,
    typename SmemOutputType>
__device__ bool processHistogramStep(
    const int* indices,
    const float* logits,
    int rowEnd,
    uint32_t& logitPattern,
    int& thresholdBinIdx,
    SmemOutputType& smemOutput,
    int* smemThresholdBinIdx,
    int* smemFinalDstIdx,
    int* smemFinalBinSize,
    int* smemFoundTopKValues,
    SmemFinalType& smemFinal,
    int stride1,
    int rowStart,
    int topK) {
  // Clear the histogram.
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Update pattern
  constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
  } else if constexpr (step == 3) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
  }

  int negativeInfinityCount = 0;
  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (useTopK512Optimization && __float_as_uint(logit) == 0xff800000u) {
        ++negativeInfinityCount;
      } else {
        atomicAdd(&smemFinal.histo.data[binIdx], 1);
      }
    }
  };

  // Distribute the elements to the histogram bins.
  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      distributeToBins(logit, idx);
    }
  }
  if constexpr (useTopK512Optimization) {
    if (negativeInfinityCount > 0) {
      const auto bin = extractBinIdx<step>(-INFINITY);
      atomicAdd(&smemFinal.histo.data[bin], negativeInfinityCount);
    }
  }
  // Make sure the histogram is ready.
  __syncthreads();

  // Reads the value of the starting position in the smemOutput array
  int lastValue = smemFoundTopKValues[0];

  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    // Read the values from SMEM.
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int binCount{0};
    binCount = smemFinal.histo.data[idx];

    // Make sure each thread has read its value.
    __syncthreads();

    // Compute the prefix sum.
    int prefixSum{0}, totalSum{0};
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

    // Update the histogram with the prefix sums.
    prefixSum += lastValue;
    totalSum += lastValue;
    smemFinal.histo.data[idx] = prefixSum;

    // Make sure the data is in shared memory.
    __syncthreads();

    // Find the last valid bin.
    bool foundThreshold = false;
    if (prefixSum < topK) {
      int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1 ? totalSum : smemFinal.histo.data[idx + 1];

      if (nextPrefixSum >= topK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - prefixSum;
        foundThreshold = true;
      }
    }

    // Early exit: if any thread found the threshold, we can skip remaining
    // rounds
    if (__syncthreads_or(foundThreshold)) {
      break;
    }

    lastValue = totalSum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The threshold bin.
  thresholdBinIdx = smemThresholdBinIdx[0];
  // Resolve a -inf cutoff with exact tie emission, keeping sort padding out.
  const bool finalBinFits =
      smemFinalBinSize[0] <= kNumFinalItems &&
      !(useTopK512Optimization && step < 3 && isPartialMatch<patternShift>(-INFINITY, logitPattern) &&
        thresholdBinIdx == extractBinIdx<step>(-INFINITY));

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      // Only write elements with binIdx < thresholdBinIdx when:
      // 1. This is step 0 and the threshold bin is small enough (no step 1)
      // 2. This is step >= 1 (where pattern matching filters correctly)
      // This prevents duplicates when step 0 and step 1 both run.
      bool shouldWriteDirectly = (step == 0 && finalBinFits) || (step >= 1);
      if (binIdx < thresholdBinIdx && shouldWriteDirectly) {
        // The element is part of the top-k selection
        int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);

        if constexpr (mergeBlocks) {
          smemOutput[dstIdx] = indices[idx];
        } else if constexpr (multipleBlocksPerRow) {
          smemOutput[dstIdx] = idx + rowStart;
          reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
        } else {
          smemOutput[dstIdx] = idx;
        }
      }
      if constexpr (step < 3) {
        // Only fill the final items for sorting if the threshold bin fits
        if (binIdx == thresholdBinIdx && finalBinFits) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          if constexpr (mergeBlocks) {
            smemFinal.items.indices[dstIdx] = indices[idx];
          } else if constexpr (multipleBlocksPerRow) {
            smemFinal.items.indices[dstIdx] = idx + rowStart;
          } else {
            smemFinal.items.indices[dstIdx] = idx;
          }
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          // The elements in the threshold bin share the same 32 bits at step 3
          int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
          if (dstIdx < topK) {
            if constexpr (mergeBlocks) {
              smemOutput[dstIdx] = indices[idx];
            } else if constexpr (multipleBlocksPerRow) {
              smemOutput[dstIdx] = idx + rowStart;
              reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
            } else {
              smemOutput[dstIdx] = idx;
            }
          }
        }
      }
    }
  };

  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, processBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      processBins(logit, idx);
    }
  }

  // Make sure the elements are in shared memory.
  __syncthreads();

  // Check if we should continue to next step
  return !finalBinFits;
}

// Follows half - 11 - 11 - 10 bit iterations
// Keep the adaptive kernel's device instantiation separate from legacy calls.
template <
    int kNumThreadsPerBlock,
    int kNumBins,
    bool useRadixSort,
    bool multipleBlocksPerRow = false,
    bool mergeBlocks = false,
    bool deviceLengthAware = false,
    bool useTopK512Optimization = false>
static __device__ void topKPerRowJob(
    const int* indices,
    const float* logits,
    int rowStart,
    int rowEnd,
    int* outIndices,
    float* outLogits,
    int stride1,
    int topK) {
  static_assert(!deviceLengthAware || multipleBlocksPerRow != mergeBlocks);
  static_assert(!useTopK512Optimization || kNumThreadsPerBlock == 1024);
  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = useTopK512Optimization ? 1024 : 2048;
  // The number of elements per thread for the final sort.
  static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
  // The class to sort the elements during the final pass.
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;
  using FinalSortTempStorage = std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
  // The class to compute the inclusive prefix-sum over the histogram.
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  // The structure to store the final items (for the final pass).
  struct FinalItems {
    // Shared memory to store the indices for the final pass.
    int indices[kNumFinalItems];
    // Shared memory to store the logits for the final pass.
    float logits[kNumFinalItems];
  };

  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };

  // Shared memory to compute the block sort.
  __shared__ union {
    FinalItems items;
    FinalSortTempStorage finalSort;
    Histogram histo;
  } smemFinal;

  // Shared memory to store the selected indices.
  // If we are processing using multiple blocks, we need to store the logits and
  // indices.
  extern __shared__ int32_t smemOutput[];

  // Shared memory to store the threshold bin.
  __shared__ int smemThresholdBinIdx[1];
  // Shared memory counter to register the candidates for the final phase.
  __shared__ int smemFinalDstIdx[1];
  // Shared memory to determine if the threshold bin fits in the final items.
  __shared__ int smemFinalBinSize[1];
  // Shared memory to keep track of the top-k values found so far by the
  // previous iterations
  __shared__ int smemFoundTopKValues[1];

  // The length of the row.
  int rowLen = rowEnd - rowStart;

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (rowLen <= topK) {
    for (int rowIt = threadIdx.x; rowIt < rowLen; rowIt += kNumThreadsPerBlock) {
      if constexpr (multipleBlocksPerRow) {
        outIndices[rowIt] = rowIt + rowStart;
        outLogits[rowIt] = logits[rowIt + rowStart];
      } else {
        outIndices[rowIt] = rowIt;
      }
    }
    for (int rowIt = rowLen + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;
      if constexpr (multipleBlocksPerRow) {
        outLogits[rowIt] = useTopK512Optimization ? -INFINITY : -FLT_MAX;
      }
    }

    return;
  }
  // Initialize values
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  __syncthreads();
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  // Step 0: Process first 11 bits of half representation
  bool continueToNextStep = processHistogramStep<
      0,
      kNumThreadsPerBlock,
      kNumBins,
      kNumFinalItems,
      multipleBlocksPerRow,
      mergeBlocks,
      useTopK512Optimization>(
      indices,
      logits,
      rowEnd,
      logitPattern,
      thresholdBinIdx,
      smemOutput,
      smemThresholdBinIdx,
      smemFinalDstIdx,
      smemFinalBinSize,
      smemFoundTopKValues,
      smemFinal,
      stride1,
      rowStart,
      topK);

  if (continueToNextStep) {
    // Step 1: Process next 11 bits
    continueToNextStep = processHistogramStep<
        1,
        kNumThreadsPerBlock,
        kNumBins,
        kNumFinalItems,
        multipleBlocksPerRow,
        mergeBlocks,
        useTopK512Optimization>(
        indices,
        logits,
        rowEnd,
        logitPattern,
        thresholdBinIdx,
        smemOutput,
        smemThresholdBinIdx,
        smemFinalDstIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        smemFinal,
        stride1,
        rowStart,
        topK);
  }

  if (continueToNextStep) {
    // Step 2: Process next 11 bits
    continueToNextStep = processHistogramStep<
        2,
        kNumThreadsPerBlock,
        kNumBins,
        kNumFinalItems,
        multipleBlocksPerRow,
        mergeBlocks,
        useTopK512Optimization>(
        indices,
        logits,
        rowEnd,
        logitPattern,
        thresholdBinIdx,
        smemOutput,
        smemThresholdBinIdx,
        smemFinalDstIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        smemFinal,
        stride1,
        rowStart,
        topK);
  }

  if (continueToNextStep) {
    // Step 3: Process last 10 bits
    processHistogramStep<
        3,
        kNumThreadsPerBlock,
        kNumBins,
        kNumFinalItems,
        multipleBlocksPerRow,
        mergeBlocks,
        useTopK512Optimization>(
        indices,
        logits,
        rowEnd,
        logitPattern,
        thresholdBinIdx,
        smemOutput,
        smemThresholdBinIdx,
        smemFinalDstIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        smemFinal,
        stride1,
        rowStart,
        topK);
  }

  if (!continueToNextStep) {
    // The histogram did not proceed to the final 10 bits, therefore we need to
    // sort the final items The logits of the elements to be sorted in the final
    // pass.
    auto insertionSort = [&]() {
      // Sorting with insertion sort
      auto baseIdx = smemFoundTopKValues[0];
      for (int i = threadIdx.x; i < smemFinalDstIdx[0]; i += kNumThreadsPerBlock) {
        int outIndex = 0;
        auto logit = smemFinal.items.logits[i];
        for (int j = 0; j < smemFinalDstIdx[0]; j++) {
          auto otherLogit = smemFinal.items.logits[j];
          if (logit < otherLogit || (logit == otherLogit && i < j)) {
            outIndex++;
          }
        }
        // Store if outIndex is in bounds
        if (outIndex + baseIdx < topK) {
          smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] = smemFinal.items.logits[i];
          }
        }
      }
    };
    if constexpr (useRadixSort) {
      if (useTopK512Optimization && smemFinalDstIdx[0] <= 128) {
        insertionSort();
      } else {
        // Sorting with radix sort
        float finalLogits[kNumFinalItemsPerThread];
        // The indices of the elements to be sorted in the final pass.
        int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
        for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
          finalLogits[ii] = useTopK512Optimization ? -INFINITY : -FLT_MAX;
          if constexpr (useTopK512Optimization) finalIndices[ii] = -1;
        }

        // Read the elements from SMEM.
#pragma unroll
        for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
          int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
          if (srcIdx < smemFinalDstIdx[0]) {
            finalLogits[ii] = smemFinal.items.logits[srcIdx];
            finalIndices[ii] = smemFinal.items.indices[srcIdx];
          }
        }
        // Make sure the shared memory has been read.
        __syncthreads();

        // Sort the elements.
        FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalLogits, finalIndices);

        // Copy the data back to the shared memory storage.
        int baseIdx = smemFoundTopKValues[0];

#pragma unroll
        for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
          int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
          int dstIdx = baseIdx + srcIdx;

          if (dstIdx < topK) {
            smemOutput[dstIdx] = finalIndices[ii];
            if constexpr (multipleBlocksPerRow) {
              reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = finalLogits[ii];
            }
          }
        }
      }
    } else {
      insertionSort();
    }
    __syncthreads();
  }

  // Store to global memory.
  for (int i = threadIdx.x; i < topK; i += kNumThreadsPerBlock) {
    if constexpr (multipleBlocksPerRow) {
      outIndices[i] = smemOutput[i];
      outLogits[i] = reinterpret_cast<float*>(smemOutput + topK)[i];
    } else {
      if (stride1 == 1) {
        // stride1 == 1 will use vectorized_process, which indexes already skip
        // the rowStart.
        outIndices[i] = smemOutput[i];
      } else {
        outIndices[i] = smemOutput[i] - rowStart;
      }
    }
  }
}

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(
    const float* logits,
    const int* rowStarts,
    const int* rowEnds,
    int* outIndices,
    int stride0,
    int stride1,
    const int topK,
    const int offsetIndex) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x + offsetIndex;

  // The range of logits within the row.
  int rowStart = rowStarts[rowIdx];
  int rowEnd = rowEnds[rowIdx];

  // Local pointers to this block
  outIndices += static_cast<int64_t>(rowIdx) * topK;
  logits += static_cast<int64_t>(rowIdx) * stride0;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort>(
      nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK);
}

template <int kNumThreadsPerBlock, bool useRadixSort, bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecode(
    const float* logits,
    const int* seqLens,
    int* outIndices,
    int stride0,
    int stride1,
    const int topK,
    int next_n,
    int seqLensIs2D = 0,
    float* outLogits = nullptr,
    const int numBlocksToMerge = 0,
    const int* indices = nullptr) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x;

  // The range of logits within the row.
  int rowStart = 0;
  int batch_idx = rowIdx / next_n;
  int next_n_idx = rowIdx % next_n;
  // seqLensIs2D=0: 1D seqLens — all rows in a batch share the same seq_len;
  //               kernel computes per-row effective length via offset.
  // seqLensIs2D=1: 2D seqLens — each logit row has its own pre-computed
  //               effective length (flat index rowIdx = b*next_n + j maps
  //               directly to seqLens[b, j] in C-contiguous layout).
  int seq_len = seqLensIs2D ? seqLens[rowIdx] : seqLens[batch_idx];
  int rowEnd = seqLensIs2D ? max(0, seq_len) : max(0, seq_len - next_n + next_n_idx + 1);

  // Local pointers to this block
  if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
    outIndices += static_cast<int64_t>(rowIdx) * topK;
  } else if constexpr (multipleBlocksPerRow) {
    const auto blockSize = rowEnd / gridDim.y;  // 16384 / 2 = 8192
    rowStart = blockSize * blockIdx.y;          // 8192 * 1 = 8192
    rowEnd = gridDim.y == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
    outIndices += static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
    outLogits += static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
  } else if constexpr (mergeBlocks) {
    rowEnd = numBlocksToMerge * topK;
    indices += static_cast<int64_t>(rowIdx) * numBlocksToMerge * topK;
    outIndices += static_cast<int64_t>(rowIdx) * topK;
  }
  logits += static_cast<int64_t>(rowIdx) * stride0;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort, multipleBlocksPerRow, mergeBlocks>(
      indices, logits, rowStart, rowEnd, outIndices, outLogits, stride1, topK);
}

struct TopKPerRowKernel {
  static void prefill(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView row_starts,
      const tvm::ffi::TensorView row_ends,
      const tvm::ffi::TensorView indices,
      const int64_t topk) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<float>().with_device(device).verify(logits);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(row_starts);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(row_ends);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(indices);
    RuntimeCheck(topk > 0, "topk must be > 0");
    RuntimeCheck(static_cast<int64_t>(K.unwrap()) == topk, "indices.width must equal topk");

    constexpr int kSortingAlgorithmThreshold = 12288;
    constexpr int kNumThreadsPerBlock = 512;
    const int num_rows = static_cast<int>(B.unwrap());
    const int stride0 = static_cast<int>(S.unwrap());
    const int stride1 = 1;
    const auto* logits_ptr = static_cast<const float*>(logits.data_ptr());
    const auto* starts_ptr = static_cast<const int*>(row_starts.data_ptr());
    const auto* ends_ptr = static_cast<const int*>(row_ends.data_ptr());
    auto* indices_ptr = static_cast<int*>(indices.data_ptr());
    const size_t smem = static_cast<size_t>(topk) * sizeof(int32_t);

    const int num_insertion = std::min(num_rows, kSortingAlgorithmThreshold);
    if (num_insertion > 0) {
      LaunchKernel(num_insertion, kNumThreadsPerBlock, device.unwrap(), smem)(
          topKPerRowPrefill<kNumThreadsPerBlock, false>,
          logits_ptr,
          starts_ptr,
          ends_ptr,
          indices_ptr,
          stride0,
          stride1,
          static_cast<int>(topk),
          0);
    }
    if (num_rows > kSortingAlgorithmThreshold) {
      const int num_radix = num_rows - kSortingAlgorithmThreshold;
      LaunchKernel(num_radix, kNumThreadsPerBlock, device.unwrap(), smem)(
          topKPerRowPrefill<kNumThreadsPerBlock, true>,
          logits_ptr,
          starts_ptr,
          ends_ptr,
          indices_ptr,
          stride0,
          stride1,
          static_cast<int>(topk),
          kSortingAlgorithmThreshold);
    }
  }

  static void decode(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::Optional<tvm::ffi::TensorView> aux_indices_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> aux_logits_opt,
      const int64_t next_n,
      const int64_t topk) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<float>().with_device(device).verify(logits);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(seq_lens);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(indices);
    RuntimeCheck(topk > 0 && next_n > 0, "topk/next_n must be > 0");
    RuntimeCheck(static_cast<int64_t>(K.unwrap()) == topk, "indices.width must equal topk");

    constexpr int kSortingAlgorithmThreshold = 12288;
    constexpr int kSplitWorkThreshold = 200 * 1000;
    constexpr int kNumThreadsPerBlock = 512;
    const int num_rows = static_cast<int>(B.unwrap());
    const int num_columns = static_cast<int>(logits.size(1));
    const int stride0 = static_cast<int>(S.unwrap());
    const int stride1 = 1;
    const int seq_lens_is_2d = seq_lens.ndim() == 2 ? 1 : 0;
    const auto* logits_ptr = static_cast<const float*>(logits.data_ptr());
    const auto* seq_ptr = static_cast<const int*>(seq_lens.data_ptr());
    auto* indices_ptr = static_cast<int*>(indices.data_ptr());
    const size_t smem = static_cast<size_t>(topk) * sizeof(int32_t);

    if (num_columns < kSortingAlgorithmThreshold) {
      LaunchKernel(num_rows, kNumThreadsPerBlock, device.unwrap(), smem)(
          topKPerRowDecode<kNumThreadsPerBlock, false>,
          logits_ptr,
          seq_ptr,
          indices_ptr,
          stride0,
          stride1,
          static_cast<int>(topk),
          static_cast<int>(next_n),
          seq_lens_is_2d,
          /*outLogits=*/static_cast<float*>(nullptr),
          /*numBlocksToMerge=*/0,
          /*indices=*/static_cast<const int*>(nullptr));
      return;
    }
    if (num_columns < kSplitWorkThreshold) {
      LaunchKernel(num_rows, kNumThreadsPerBlock, device.unwrap(), smem)(
          topKPerRowDecode<kNumThreadsPerBlock, true>,
          logits_ptr,
          seq_ptr,
          indices_ptr,
          stride0,
          stride1,
          static_cast<int>(topk),
          static_cast<int>(next_n),
          seq_lens_is_2d,
          /*outLogits=*/static_cast<float*>(nullptr),
          /*numBlocksToMerge=*/0,
          /*indices=*/static_cast<const int*>(nullptr));
      return;
    }

    // Long rows: split then merge. Caller must supply aux buffers.
    RuntimeCheck(
        aux_indices_opt.has_value() && aux_logits_opt.has_value(),
        "long-row top_k_per_row_decode requires aux_indices and aux_logits");
    constexpr int multipleBlocksPerRowConfig = 10;
    auto aux_indices = aux_indices_opt.value();
    auto aux_logits = aux_logits_opt.value();
    TensorMatcher({B, multipleBlocksPerRowConfig, K}).with_dtype<int32_t>().with_device(device).verify(aux_indices);
    TensorMatcher({B, multipleBlocksPerRowConfig, K}).with_dtype<float>().with_device(device).verify(aux_logits);

    dim3 grid(num_rows, multipleBlocksPerRowConfig);
    const size_t smem_split = 2 * static_cast<size_t>(topk) * sizeof(int32_t);
    LaunchKernel(grid, kNumThreadsPerBlock, device.unwrap(), smem_split)(
        topKPerRowDecode<kNumThreadsPerBlock, true, true>,
        logits_ptr,
        seq_ptr,
        static_cast<int*>(aux_indices.data_ptr()),
        stride0,
        stride1,
        static_cast<int>(topk),
        static_cast<int>(next_n),
        seq_lens_is_2d,
        static_cast<float*>(aux_logits.data_ptr()),
        /*numBlocksToMerge=*/0,
        /*indices=*/static_cast<const int*>(nullptr));

    constexpr int kNumThreadsPerBlockMerge = 1024;
    LaunchKernel(num_rows, kNumThreadsPerBlockMerge, device.unwrap(), smem)(
        topKPerRowDecode<kNumThreadsPerBlockMerge, true, false, true>,
        static_cast<const float*>(aux_logits.data_ptr()),
        seq_ptr,
        indices_ptr,
        multipleBlocksPerRowConfig * static_cast<int>(topk),
        1,
        static_cast<int>(topk),
        static_cast<int>(next_n),
        seq_lens_is_2d,
        nullptr,
        multipleBlocksPerRowConfig,
        static_cast<const int*>(aux_indices.data_ptr()));
  }
};

}  // namespace topk_per_row
}  // namespace sglang
