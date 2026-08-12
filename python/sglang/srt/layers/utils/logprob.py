from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.sampling_params import TOP_K_ALL

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata


class LogprobStage(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclasses.dataclass
class InputLogprobsResult:
    input_token_logprobs: torch.Tensor
    input_top_logprobs_val: Optional[List] = None
    input_top_logprobs_idx: Optional[List] = None
    input_token_ids_logprobs_val: Optional[List] = None
    input_token_ids_logprobs_idx: Optional[List] = None


def get_top_logprobs_raw(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    max_k = max(top_logprobs_nums)
    values, indices = logprobs.topk(max_k, dim=-1)
    if not no_copy_to_cpu:
        values = values.tolist()
        indices = indices.tolist()

    top_logprobs_val = []
    top_logprobs_idx = []

    if stage == LogprobStage.DECODE:
        for i, k in enumerate(top_logprobs_nums):
            top_logprobs_val.append(values[i][:k])
            top_logprobs_idx.append(indices[i][:k])
    else:
        pt = 0
        for k, pruned_len in zip(top_logprobs_nums, extend_logprob_pruned_lens_cpu):
            if pruned_len <= 0:
                top_logprobs_val.append([])
                top_logprobs_idx.append([])
                continue

            top_logprobs_val.append([values[pt + j][:k] for j in range(pruned_len)])
            top_logprobs_idx.append([indices[pt + j][:k] for j in range(pruned_len)])
            pt += pruned_len

    return top_logprobs_val, top_logprobs_idx


def get_top_logprobs_prefill(
    all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata
):
    return get_top_logprobs_raw(
        all_logprobs,
        logits_metadata.top_logprobs_nums,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
    )


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    no_copy_to_cpu: bool = False,
):
    return get_top_logprobs_raw(
        logprobs,
        top_logprobs_nums,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def _top_p_filter_rows(
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_p_sampling: bool,
    need_min_p_sampling: bool,
    request_mask: torch.Tensor,
) -> torch.Tensor:
    """Rows that were requested AND actually have a top-k/top-p/min-p filter."""
    row_has_filter = top_ks != TOP_K_ALL
    if need_top_p_sampling:
        row_has_filter = row_has_filter | (top_ps != 1.0)
    if need_min_p_sampling:
        row_has_filter = row_has_filter | (min_ps > 0)
    return request_mask & row_has_filter


def _top_p_keep_mask_sorted(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_p_sampling: bool,
    need_min_p_sampling: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean nucleus keep-mask in descending-prob order, plus the sort indices.

    Reproduces SGLang's sampler truncation (rank < top_k, cumulative prob within
    top_p, prob >= top1 * min_p) so replay sees the exact set the sampler keeps.
    """
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    ranks = torch.arange(probs_sort.shape[-1], device=probs_sort.device).view(1, -1)
    keep = ranks < top_ks.view(-1, 1)
    if need_top_p_sampling:
        keep &= (torch.cumsum(probs_sort, dim=-1) - probs_sort) <= top_ps.view(-1, 1)
    if need_min_p_sampling:
        keep &= probs_sort >= (probs_sort[:, 0] * min_ps).view(-1, 1)
    return keep, probs_idx


def renorm_logprob_over_top_p(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_p_sampling: bool,
    need_min_p_sampling: bool,
    request_mask: torch.Tensor,
    force_keep_token_ids: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    rows = _top_p_filter_rows(
        top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling, request_mask
    )
    if not bool(rows.any().item()):
        return None

    keep, probs_idx = _top_p_keep_mask_sorted(
        probs, top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling
    )
    # Scatter the keep-mask back to vocab order so we renormalize directly over
    # vocab ids (and can force-keep specific token ids).
    keep_vocab = torch.empty_like(keep)
    keep_vocab.scatter_(-1, probs_idx, keep)

    if force_keep_token_ids is not None:
        # Force-keep the sampled/accepted token so its renormalized logprob is
        # finite even when SGLang's sampling kernel (e.g. flashinfer) keeps a
        # boundary token that this torch nucleus drops. This matches the trainer,
        # which also force-keeps the target token before renormalizing, so the
        # rollout and training denominators are both ``nucleus ∪ {token}``.
        # Non-filter rows are overwritten by the ``torch.where`` below, so
        # force-keeping every row is harmless and avoids a row gather.
        row_idx = torch.arange(keep_vocab.shape[0], device=keep_vocab.device)
        keep_vocab[row_idx, force_keep_token_ids] = True

    kept_probs = probs * keep_vocab
    kept_probs = kept_probs / kept_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(rows.view(-1, 1), torch.log(kept_probs), torch.log(probs))


def get_top_p_token_ids_from_probs(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_top_p_sampling: bool,
    need_min_p_sampling: bool,
    request_mask: torch.Tensor,
) -> Optional[List[Optional[torch.Tensor]]]:
    rows = _top_p_filter_rows(
        top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling, request_mask
    )
    if not bool(rows.any().item()):
        return None

    keep, probs_idx = _top_p_keep_mask_sorted(
        probs, top_ks, top_ps, min_ps, need_top_p_sampling, need_min_p_sampling
    )
    return [
        probs_idx[i][keep[i]].to(torch.int32) if bool(rows[i].item()) else None
        for i in range(probs.shape[0])
    ]


def get_token_ids_logprobs_raw(
    logprobs: torch.Tensor,
    token_ids_logprobs_list: List[Optional[List[int]]],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    vals, idxs = [], []
    if stage == LogprobStage.DECODE:
        for i, token_ids in enumerate(token_ids_logprobs_list):
            if token_ids is None:
                vals.append([])
                idxs.append([])
            else:
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                    logprobs.device, non_blocking=True
                )
                row = logprobs[i, token_ids_tensor]
                vals.append(row if no_copy_to_cpu else row.tolist())
                idxs.append(token_ids)
    else:  # prefill
        pt = 0
        for i, (token_ids, pruned_len) in enumerate(
            zip(token_ids_logprobs_list, extend_logprob_pruned_lens_cpu)
        ):
            if pruned_len <= 0:
                vals.append([])
                idxs.append([])
                continue
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                logprobs.device, non_blocking=True
            )
            pos_logprobs = logprobs[pt : pt + pruned_len, token_ids_tensor]
            vals.append(pos_logprobs if no_copy_to_cpu else pos_logprobs.tolist())
            idxs.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len
    return vals, idxs


def get_token_ids_logprobs_prefill(
    all_logprobs, logits_metadata: LogitsMetadata, no_copy_to_cpu=False
):
    return get_token_ids_logprobs_raw(
        all_logprobs,
        logits_metadata.token_ids_logprobs,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_token_ids_logprobs(logprobs, token_ids_logprobs, no_copy_to_cpu=False):
    return get_token_ids_logprobs_raw(
        logprobs,
        token_ids_logprobs,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_top_logprobs_chunk(
    logprobs: torch.Tensor,
    logits_metadata: LogitsMetadata,
    top_k_nums: List[int],
    pruned_lens: List[int],
    input_top_logprobs_val: List,
    input_top_logprobs_idx: List,
    split_pruned_len: int,
) -> int:
    """Get top-k logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        logits_metadata: Metadata containing top-k and pruned length info
        top_k_nums: List of top-k numbers for each sequence
        pruned_lens: List of pruned lengths for each sequence
        input_top_logprobs_val: List to store top-k logprob values
        input_top_logprobs_idx: List to store top-k token indices
        split_pruned_len: Length of pruned tokens from previous chunk

    Returns:
        int: Number of remaining tokens to process in next chunk
    """
    # No sequences in the chunk
    if logprobs.shape[0] == 0:
        return 0

    max_k = max(logits_metadata.top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    pt = 0
    next_split_pruned_len = 0
    for n, (k, pruned_len) in enumerate(zip(top_k_nums, pruned_lens)):
        if n == 0:
            # For the first sequence, adjust the pruned length
            pruned_len -= split_pruned_len
        else:
            # After the first sequence, no split in the middle
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no top-k logprobs to process
            input_top_logprobs_val.append([])
            input_top_logprobs_idx.append([])
            continue

        # Get the top-k logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= len(values):
                next_split_pruned_len = split_pruned_len + j
                break
            # Append the top-k logprobs
            val.append(values[pt + j][:k])
            idx.append(indices[pt + j][:k])

        # Append or extend based on whether the sequence was split across chunks
        if len(val) > 0:
            if split_pruned_len > 0:
                input_top_logprobs_val[-1].extend(val)
                input_top_logprobs_idx[-1].extend(idx)
            else:
                input_top_logprobs_val.append(val)
                input_top_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def get_token_ids_logprobs_chunk(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[int],
    pruned_lens: List[int],
    input_token_ids_logprobs_val: List,
    input_token_ids_logprobs_idx: List,
    split_pruned_len: int = 0,
):
    """Get token_ids logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        logits_metadata: Metadata containing token IDs and pruned length info
        token_ids_logprobs: List of token IDs for each sequence
        pruned_lens: List of pruned lengths for each sequence
        input_token_ids_logprobs_val: List to store token logprob values
        input_token_ids_logprobs_idx: List to store token indices
        split_pruned_len: Length of pruned tokens from previous chunk

    Returns:
        int: Number of remaining tokens to process in next chunk
    """

    # No sequences in the chunk
    if logprobs.shape[0] == 0:
        return 0

    pt = 0
    next_split_pruned_len = 0
    for n, (token_ids, pruned_len) in enumerate(
        zip(
            token_ids_logprobs,
            pruned_lens,
        )
    ):
        # Adjust pruned length for first sequence
        if n == 0:
            pruned_len -= split_pruned_len
        else:
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no token ids logprobs to process
            input_token_ids_logprobs_val.append([])
            input_token_ids_logprobs_idx.append([])
            continue

        # Get the token ids logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= logprobs.shape[0]:
                next_split_pruned_len = split_pruned_len + j
                break
            if token_ids is not None:
                val.append(logprobs[pt + j, token_ids].tolist())
                idx.append(token_ids)

        # Append or extend based on whether the sequence was split across chunks
        if len(val) > 0:
            if split_pruned_len > 0:
                input_token_ids_logprobs_val[-1].extend(val)
                input_token_ids_logprobs_idx[-1].extend(idx)
            else:
                input_token_ids_logprobs_val.append(val)
                input_token_ids_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def compute_spec_v2_logprobs(
    batch,
    logits_output,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    speculative_num_steps: int,
):
    """Compute logprobs for accepted tokens after spec v2 verify sampling.

    Gathers logits at accepted positions, applies log_softmax (temperature-scaled
    if not greedy), and populates logits_output.next_token_logprobs plus optional
    top-k / token-ids / top-p replay metadata so they flow through copy_to_cpu().
    """
    bs = len(batch.seq_lens)
    max_accept = speculative_num_steps + 1
    device = predict.device

    flat_accept_idx = accept_index.long().reshape(-1)
    gathered_logits = logits_output.next_token_logits[flat_accept_idx]
    temperatures = None

    if batch.sampling_info.is_all_greedy or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        gathered_logprobs = torch.nn.functional.log_softmax(gathered_logits, dim=-1)
    else:
        temperatures = torch.repeat_interleave(
            batch.sampling_info.temperatures,
            max_accept,
            dim=0,
        )
        gathered_logprobs = torch.nn.functional.log_softmax(
            gathered_logits / temperatures, dim=-1
        )
    gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)

    accepted_token_ids = predict[flat_accept_idx]
    token_logprobs = gathered_logprobs[
        torch.arange(bs * max_accept, device=device),
        accepted_token_ids.long(),
    ]
    logits_output.next_token_logprobs = token_logprobs.reshape(bs, max_accept)

    if batch.sampling_info.need_return_top_p_token_ids:
        valid_accept_mask = (
            torch.arange(max_accept, device=device).view(1, -1)
            < accept_lens.view(-1, 1)
        ).reshape(-1)
        request_mask = (
            torch.repeat_interleave(
                batch.sampling_info.return_top_p_token_ids, max_accept
            )
            & valid_accept_mask
        )

        if batch.sampling_info.is_all_greedy:
            logits_output.next_token_top_p_token_ids = [
                (
                    accepted_token_ids[i : i + 1].to(torch.int32)
                    if bool(request_mask[i].item())
                    else None
                )
                for i in range(bs * max_accept)
            ]
        else:
            if temperatures is None:
                temperatures = torch.repeat_interleave(
                    batch.sampling_info.temperatures,
                    max_accept,
                    dim=0,
                )
            probs = torch.softmax(gathered_logits / temperatures, dim=-1)
            expanded_top_ks = torch.repeat_interleave(
                batch.sampling_info.top_ks, max_accept
            )
            expanded_top_ps = torch.repeat_interleave(
                batch.sampling_info.top_ps, max_accept
            )
            expanded_min_ps = torch.repeat_interleave(
                batch.sampling_info.min_ps, max_accept
            )
            top_p_token_ids = get_top_p_token_ids_from_probs(
                probs=probs,
                top_ks=expanded_top_ks,
                top_ps=expanded_top_ps,
                min_ps=expanded_min_ps,
                need_top_p_sampling=batch.sampling_info.need_top_p_sampling,
                need_min_p_sampling=False,
                request_mask=request_mask,
            )
            if top_p_token_ids is not None:
                logits_output.next_token_top_p_token_ids = top_p_token_ids

            renorm_logprobs = renorm_logprob_over_top_p(
                probs=probs,
                top_ks=expanded_top_ks,
                top_ps=expanded_top_ps,
                min_ps=expanded_min_ps,
                need_top_p_sampling=batch.sampling_info.need_top_p_sampling,
                need_min_p_sampling=False,
                request_mask=request_mask,
                # Force-keep the accepted token: a small fraction of
                # speculatively accepted tokens land outside their own top-p
                # nucleus, which would give a -inf renormalized logprob.
                # Force-keeping makes the denominator ``nucleus ∪ {accepted}``,
                # matching the trainer which also force-keeps the target token,
                # so these tokens stay finite and on-policy.
                force_keep_token_ids=accepted_token_ids.long(),
            )
            if renorm_logprobs is not None:
                idx = torch.arange(bs * max_accept, device=device)
                renorm_token_logprobs = renorm_logprobs[idx, accepted_token_ids.long()]
                renorm_token_logprobs.clamp_(
                    min=torch.finfo(renorm_token_logprobs.dtype).min
                )
                logits_output.next_token_logprobs = renorm_token_logprobs.reshape(
                    bs, max_accept
                )

    if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):
        top_logprobs_nums_expanded = [
            num for num in batch.top_logprobs_nums for _ in range(max_accept)
        ]
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(
            gathered_logprobs, top_logprobs_nums_expanded, no_copy_to_cpu=True
        )

    if batch.token_ids_logprobs and any(
        x is not None for x in batch.token_ids_logprobs
    ):
        token_ids_logprobs_expanded = [
            ids for ids in batch.token_ids_logprobs for _ in range(max_accept)
        ]
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            gathered_logprobs, token_ids_logprobs_expanded, no_copy_to_cpu=True
        )
