"""KPool compression and query metadata for Prefill-CP (interleave / zigzag)."""

import torch

from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.interleave import (
    interleave_rows_per_request,
    is_interleave_extend,
)
from sglang.srt.layers.cp.utils import is_dsa_cp_extend
from sglang.srt.runtime_context import get_parallel


def local_query_inputs(forward_batch, metadata):
    """Compression covers full requests; logits cover only this rank's queries."""
    if not is_dsa_cp_extend(forward_batch):
        return {}
    parallel = get_parallel()
    req_pool_indices = forward_batch.req_pool_indices
    # PR43277_ZIGZAG_DSA_SEGMENTS: zigzag segment list may repeat request ids
    # (early then late). Prefer metadata segment lengths already produced by
    # shard_per_request; rebuild matching req-pool rows (with duplicates).
    extend_list = metadata.dsa_extend_seq_lens_list
    if is_interleave_extend(forward_batch):
        counts = interleave_rows_per_request(
            forward_batch.extend_seq_lens_cpu,
            parallel.attn_cp_rank,
            parallel.attn_cp_size,
        )
        req_ids = [i for i, count in enumerate(counts) if count]
    else:
        # Match ZigzagCPStrategy.shard_per_request early-all / late-all order.
        cp_rank = parallel.attn_cp_rank
        cp_size = parallel.attn_cp_size
        req_ids = []
        for block in (cp_rank, 2 * cp_size - 1 - cp_rank):
            for request, length in enumerate(forward_batch.extend_seq_lens_cpu):
                base, remainder = divmod(int(length), 2 * cp_size)
                count = base + int(block < remainder)
                if count:
                    req_ids.append(request)
        if extend_list is not None and len(req_ids) != len(extend_list):
            raise RuntimeError(
                f"zigzag local_query_inputs segment mismatch: "
                f"req_ids={len(req_ids)} extend_list={len(extend_list)}"
            )
    indices = torch.tensor(
        req_ids,
        dtype=torch.long,
        pin_memory=req_pool_indices.is_cuda,
    )
    if req_pool_indices.is_cuda:
        indices = indices.to(device=req_pool_indices.device, non_blocking=True)

    # CP_PATH_DUMP_V1_LOCAL_QUERY
    _ret = dict(
        local_real_page_table=metadata.real_page_table,
        local_seqlens_expanded=metadata.dsa_seqlens_expanded,
        local_extend_seq_lens_cpu=extend_list,
        local_seq_lens_cpu=metadata.indexer_seq_lens_cpu.tolist(),
        local_req_pool_indices=req_pool_indices.index_select(0, indices),
    )
    try:
        from sglang.srt.layers.cp.cp_path_dump import dump_stage

        dump_stage(
            "local_query",
            {
                "req_ids": list(req_ids),
                "extend_list": list(extend_list) if extend_list is not None else None,
                "local_seq_lens_cpu": list(_ret["local_seq_lens_cpu"]),
                "local_req_pool_indices": _ret["local_req_pool_indices"],
                "local_seqlens_expanded": _ret["local_seqlens_expanded"],
                "local_real_page_table_rows": int(
                    _ret["local_real_page_table"].shape[0]
                )
                if _ret["local_real_page_table"] is not None
                else None,
            },
            n_rows=int(_ret["local_seqlens_expanded"].numel())
            if _ret["local_seqlens_expanded"] is not None
            else 0,
        )
    except Exception as _e_cpd:
        print("[CP_PATH_DUMP] local_query fail %r" % (_e_cpd,), flush=True)
    return _ret


def materialize_compression_inputs(key, score, positions, forward_batch):
    """Replicate ordered keys/scores before pooling across adjacent tokens.

    Queries and head gates stay local. Every rank writes its replicated index
    cache and tail state, so subsequent decode needs no layer-owner protocol.

    Both interleave and zigzag gather via ``get_cp_strategy().gather_kv_cache``,
    which restores global chronological token order before kpool.
    """
    if not is_dsa_cp_extend(forward_batch):
        return key, score, positions
    width = key.shape[-1]
    # SGLANG_DSA_L3_DUMP: layout probe before gather_kv_cache
    try:
        import os as _os

        tag = _os.environ.get("SGLANG_DSA_L3_DUMP")
        if tag:
            import torch as _torch
            import torch.distributed as _dist

            from sglang.srt.layers.cp.interleave import is_interleave_extend as _iil
            from sglang.srt.layers.cp.zigzag import is_zigzag_extend as _izz

            meta = getattr(forward_batch, "attn_cp_metadata", None)
            rank = (
                int(_dist.get_rank())
                if _dist.is_available() and _dist.is_initialized()
                else 0
            )
            root = _os.environ.get(
                "SGLANG_DSA_L3_DIR",
                "/lustre/projects/polyullm/mingfa/tmp/kcp_overlay_20261111/dsa_l3",
            )
            _os.makedirs(root, exist_ok=True)
            cnt_p = _os.path.join(root, "%s_rank%d_hits.cnt" % (tag, rank))
            try:
                hits = int(open(cnt_p).read().strip() or "0")
            except Exception:
                hits = 0
            max_hits = int(_os.environ.get("SGLANG_DSA_L3_MAX_HITS", "2"))
            min_rows = int(_os.environ.get("SGLANG_DSA_L3_MIN_ROWS", "128"))
            if hits < max_hits and key.shape[0] >= min_rows:
                open(cnt_p, "w").write(str(hits + 1))
                per_rank = (
                    getattr(meta, "per_rank_actual_token", None) if meta else None
                )
                pos_rows = -1
                if positions is not None and hasattr(positions, "shape"):
                    pos_rows = int(
                        positions.shape[-1]
                        if positions.dim() > 1
                        else positions.shape[0]
                    )
                fb_pos = getattr(forward_batch, "positions", None)
                info = {
                    "tag": tag,
                    "rank": rank,
                    "key_rows_before": int(key.shape[0]),
                    "score_rows_before": int(score.shape[0]),
                    "positions_rows": pos_rows,
                    "fb_positions_rows": int(fb_pos.shape[0])
                    if fb_pos is not None
                    else -1,
                    "input_ids_rows": int(forward_batch.input_ids.shape[0])
                    if getattr(forward_batch, "input_ids", None) is not None
                    else -1,
                    "is_zigzag": bool(_izz(forward_batch)),
                    "is_interleave": bool(_iil(forward_batch)),
                    "meta_type": type(meta).__name__ if meta is not None else None,
                    "per_rank_actual_token": list(per_rank)
                    if per_rank is not None
                    else None,
                    "per_rank_logical_token": list(
                        getattr(meta, "per_rank_logical_token", None) or []
                    )
                    if meta
                    else None,
                }
                packed_probe = _torch.cat((key, score), dim=-1)
                gathered = get_cp_strategy().gather_kv_cache(
                    packed_probe, forward_batch
                )
                info["key_rows_after"] = int(gathered.shape[0])
                info["gather_delta"] = int(gathered.shape[0]) - int(key.shape[0])
                info["key_before_norm0"] = float(key[0].float().norm())
                info["key_before_norm_last"] = float(key[-1].float().norm())
                info["key_after_norm0"] = float(gathered[0, :width].float().norm())
                # CP_PATH_DUMP_V3_COMPRESS: save actual tensors too
                key_before = key.detach()
                score_before = score.detach()
                key_after = gathered[..., :width].contiguous().detach()
                score_after = gathered[..., width:].contiguous().detach()
                info_tensors = {
                    **info,
                    "key_before": key_before.float().cpu(),
                    "score_before": score_before.float().cpu(),
                    "key_after": key_after.float().cpu(),
                    "score_after": score_after.float().cpu(),
                    "positions_before": positions.detach().cpu()
                    if positions is not None
                    else None,
                    "positions_after": forward_batch.positions.detach().cpu()
                    if getattr(forward_batch, "positions", None) is not None
                    else None,
                }
                path = _os.path.join(
                    root, "%s_compress_rank%d_hit%d.pt" % (tag, rank, hits)
                )
                _torch.save(info_tensors, path)
                try:
                    from sglang.srt.layers.cp.cp_path_dump import dump_stage

                    if _os.environ.get("SGLANG_CP_PATH_DUMP"):
                        dump_stage(
                            "compress",
                            {
                                "key_before": key_before,
                                "score_before": score_before,
                                "key_after": key_after,
                                "score_after": score_after,
                                "key_rows_before": int(key_before.shape[0]),
                                "key_rows_after": int(key_after.shape[0]),
                                "gather_delta": int(key_after.shape[0])
                                - int(key_before.shape[0]),
                            },
                            n_rows=int(key_before.shape[0]),
                        )
                except Exception as _e_cpd:
                    print("[CP_PATH_DUMP] compress fail %r" % (_e_cpd,), flush=True)
                print("[DSA_L3] compress " + str(info), flush=True)
                return (
                    gathered[..., :width].contiguous(),
                    gathered[..., width:].contiguous(),
                    forward_batch.positions,
                )
    except Exception as _e:
        print("[DSA_L3] compress_dump_fail %r" % (_e,), flush=True)
    packed = torch.cat((key, score), dim=-1)
    packed = get_cp_strategy().gather_kv_cache(packed, forward_batch)
    return (
        packed[..., :width].contiguous(),
        packed[..., width:].contiguous(),
        forward_batch.positions,
    )
