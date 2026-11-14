# SPDX-License-Identifier: Apache-2.0
"""Model integration and topology checks for CP-TP group sharing: prefill CP
for hybrid linear-attention models where the CP group is the TP group."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolved_view,
    resolving_view,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

# Architectures whose model code implements CP-TP group sharing: the CP group
# is the TP group, the residual stream / attention / indexer are CP-sharded,
# MoE and linear attention keep their TP partition (the linear attention
# gathers the sequence where its recurrence needs it). Model ports register
# here; every other model keeps the ordinary prefill CP path.
_SUPPORTED_MODELS: set[str] = {"Glm5NextForConditionalGeneration"}


def resolve_cp_tp_group_sharing(server_args: Any, model: Any) -> None:
    cfg, view = resolving_view(server_args), resolved_view(server_args)
    if (
        not cfg.enable_prefill_cp
        or view.attn_cp_size <= 1
        or model.hf_config.architectures[0] not in _SUPPORTED_MODELS
    ):
        return
    # Attention DP or attention TP would leave the CP group narrower than TP.
    if view.attn_cp_size != cfg.tp_size:
        raise ValueError(
            "CP-TP group sharing requires --attn-cp-size == --tp-size, got "
            f"attn_cp_size={view.attn_cp_size}, tp_size={cfg.tp_size}."
        )
    arch = model.hf_config.architectures[0]
    is_glm5_next = arch == "Glm5NextForConditionalGeneration"
    # Default Flash path stays interleave. Zigzag is an experimental DSA/KCP
    # path (vLLM #59303/#59304 style): opt in with --cp-strategy zigzag.
    if is_glm5_next:
        allowed = ("interleave", "zigzag")
        if cfg.cp_strategy not in allowed:
            raise ValueError(
                "CP-TP group sharing for Glm5Next requires --cp-strategy "
                f"interleave|zigzag, got {cfg.cp_strategy!r}."
            )
    else:
        if cfg.cp_strategy != "zigzag":
            raise ValueError("CP-TP group sharing requires --cp-strategy zigzag.")

    # This hook runs before speculative defaults and NEXTN alias resolution.
    # Native MTP keeps the target's CP/TP placement; other draft architectures
    # have their own topology and model-boundary contracts.
    # Zigzag DSA/KCP first cut rejects speculative decoding (same as vLLM draft).
    # #49089: interleave + native MTP/EAGLE topk1 is allowed.
    if cfg.speculative_algorithm is not None:
        if cfg.cp_strategy == "zigzag":
            raise ValueError(
                "Glm5Next zigzag Prefill-CP does not support speculative decoding yet."
            )
        if (
            not is_glm5_next
            or cfg.speculative_algorithm.upper() not in ("EAGLE", "NEXTN")
            or cfg.speculative_draft_model_path not in (None, cfg.model_path)
            or cfg.speculative_eagle_topk not in (None, 1)
        ):
            raise ValueError(
                "CP/TP group sharing only supports native MTP with top-k 1."
            )

    # DCP is intentionally allowed: persistent MLA KV uses the existing
    # DCP stripe write (`set_mla_kv_buffer_dcp_sharded_*`) and CP extend
    # temporarily materializes full KV into `dcp_kv_buffer` (#36990 CP-DCP).
    # Prefill still shards compute via CP; KDA keeps SumGroup.TP under sharing.
    unsupported = {
        "enable-dsa-cache-layer-split": cfg.enable_dsa_cache_layer_split,
        "enable-mixed-chunk": cfg.enable_mixed_chunk,
        "moe-dp-size > 1": cfg.moe_dp_size > 1,
    }
    # #43774: allow PD prefill + PCP for Glm5Next (mooncake).
    # Decode PD remains blocked; non-Glm5Next still forbids all disaggregation.
    if is_glm5_next:
        if cfg.disaggregation_mode != "null":
            unsupported |= {
                "disaggregation-mode decode": cfg.disaggregation_mode == "decode",
                "disaggregation-transfer-backend != mooncake": (
                    cfg.disaggregation_transfer_backend != "mooncake"
                ),
                "pp-size > 1": cfg.pp_size > 1,
                "dp-size > 1": cfg.dp_size > 1,
                "ep-size > 1": view.ep_size > 1,
            }
    else:
        unsupported["disaggregation-mode"] = cfg.disaggregation_mode != "null"
    for flag, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"--{flag} is not supported with CP-TP group sharing.")

    linear_config = mambaish_config(model)
    if getattr(linear_config, "linear_num_key_heads", None) is not None:
        heads = {
            name: getattr(linear_config, name)
            for name in ("linear_num_key_heads", "linear_num_value_heads")
        }
    else:
        heads = {"num_heads": linear_config.linear_attn_config["num_heads"]}
    # Linear attention keeps its TP head partition over the whole TP group.
    for name, count in heads.items():
        if count % cfg.tp_size:
            raise ValueError(
                f"{name}={count} must be divisible by --tp-size={cfg.tp_size}."
            )

    declare_resolution(
        server_args,
        "resolve_cp_tp_group_sharing",
        enable_cp_tp_group_sharing=True,
        # The sequence collectives have batch-dependent sizes.
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )
