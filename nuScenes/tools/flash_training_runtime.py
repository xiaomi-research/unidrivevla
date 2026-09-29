"""Training-time TF32/SDPA and selective torch.compile setup."""

from __future__ import annotations

from typing import Iterable, List, Set, Tuple

import torch
from torch import nn


def _log(logger, message: str) -> None:
    if logger is None:
        print(message)
    else:
        logger.info(message)


def _planning_head(model: nn.Module) -> nn.Module:
    module = getattr(model, "module", model)
    head = getattr(module, "planning_head", None)
    if head is None:
        raise AttributeError("training_runtime requires model.planning_head")
    return head


def _unique_named_modules(
    named_modules: Iterable[Tuple[str, nn.Module]],
) -> List[Tuple[str, nn.Module]]:
    result: List[Tuple[str, nn.Module]] = []
    seen: Set[int] = set()
    for name, module in named_modules:
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        result.append((name, module))
    return result


def _compile_targets(head: nn.Module, target_names: Iterable[str]):
    target_names = set(target_names)
    supported = {
        "visual",
        "action_expert",
        "action_query_decoder",
        "unified_decoder_refine",
        "unified_decoder_map_ego_refine",
        "unified_decoder_ego_refine",
        "unified_decoder_anchor_encoders",
        "unified_decoder_gnn",
    }
    unknown = target_names - supported
    if unknown:
        raise ValueError(f"Unknown training compile targets: {sorted(unknown)}")

    matched = {name: 0 for name in target_names}
    qwen = head.qwen3_vl_with_expert
    decoder = head.unified_decoder
    candidates: List[Tuple[str, nn.Module]] = []

    if "visual" in target_names:
        candidates.append(("visual", qwen.qwen3_vl.visual))
        matched["visual"] += 1
    if "action_expert" in target_names:
        candidates.append(("action_expert", qwen.qwen3_action_expert))
        matched["action_expert"] += 1

    if "action_query_decoder" in target_names:
        candidates.append(("action_query_decoder", head.action_query_decoder))
        matched["action_query_decoder"] += 1

    if "unified_decoder_refine" in target_names:
        for task in ("det", "map", "ego", "motion"):
            modules = getattr(decoder, f"{task}_refine", None)
            if modules is not None:
                additions = [
                    (f"unified_decoder.{task}_refine.{index}", module)
                    for index, module in enumerate(modules)
                ]
                candidates.extend(additions)
                matched["unified_decoder_refine"] += len(additions)

    if "unified_decoder_map_ego_refine" in target_names:
        for task in ("map", "ego"):
            modules = getattr(decoder, f"{task}_refine", None)
            if modules is not None:
                additions = [
                    (f"unified_decoder.{task}_refine.{index}", module)
                    for index, module in enumerate(modules)
                ]
                candidates.extend(additions)
                matched["unified_decoder_map_ego_refine"] += len(additions)

    if "unified_decoder_ego_refine" in target_names:
        modules = getattr(decoder, "ego_refine", None)
        if modules is not None:
            additions = [
                (f"unified_decoder.ego_refine.{index}", module)
                for index, module in enumerate(modules)
            ]
            candidates.extend(additions)
            matched["unified_decoder_ego_refine"] += len(additions)

    if "unified_decoder_anchor_encoders" in target_names:
        for task in ("det", "map", "ego", "motion"):
            module = getattr(decoder, f"{task}_anchor_encoder", None)
            if module is not None:
                candidates.append(
                    (f"unified_decoder.{task}_anchor_encoder", module)
                )
                matched["unified_decoder_anchor_encoders"] += 1

    if "unified_decoder_gnn" in target_names:
        gnn_ops = {"gnn", "temp_gnn", "inter_gnn"}
        for index, (op, module) in enumerate(
            zip(decoder.operation_order, decoder.layers)
        ):
            if op in gnn_ops and module is not None:
                candidates.append(
                    (f"unified_decoder.layers.{index}:{op}", module)
                )
                matched["unified_decoder_gnn"] += 1

    unmatched = sorted(name for name, count in matched.items() if count == 0)
    if unmatched:
        raise RuntimeError(
            "Training compile target groups matched no modules: "
            f"{unmatched}"
        )
    return _unique_named_modules(candidates)


def configure_training_runtime(model: nn.Module, cfg, logger=None) -> None:
    runtime = cfg.get("training_runtime", None)
    if not runtime or not runtime.get("enabled", False):
        return

    allow_tf32 = bool(runtime.get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(
        runtime.get("cudnn_benchmark", False)
    )
    matmul_precision = runtime.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(matmul_precision)

    head = _planning_head(model)
    attention_impl = runtime.get("attention_impl", None)
    if attention_impl:
        if attention_impl not in {"eager", "sdpa", "flex"}:
            raise ValueError(
                f"Unsupported training attention_impl={attention_impl}"
            )
        head.attn_implementation = attention_impl
        head.qwen3_vl_with_expert._vla_attn_impl = attention_impl
        for module in (
            head.qwen3_vl_with_expert.qwen3_vl.language_model,
            head.qwen3_vl_with_expert.qwen3_perception_expert,
            head.qwen3_vl_with_expert.qwen3_action_expert,
        ):
            module.config._attn_implementation = attention_impl

    _log(
        logger,
        "[TrainingRuntime] "
        f"TF32={allow_tf32}, matmul_precision={matmul_precision}, "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark}, "
        f"attention={attention_impl or 'config-default'}",
    )

    compile_cfg = runtime.get("compile", {})
    if not compile_cfg.get("enabled", False):
        return
    if not hasattr(nn.Module, "compile"):
        raise RuntimeError(
            "This training config requires torch.nn.Module.compile()"
        )

    import torch._dynamo.config as dynamo_config

    dynamo_cache_size_limit = int(
        compile_cfg.get("dynamo_cache_size_limit", 8)
    )
    dynamo_config.cache_size_limit = dynamo_cache_size_limit

    backend = compile_cfg.get("backend", "inductor")
    mode = compile_cfg.get("mode", "reduce-overhead")
    fullgraph = bool(compile_cfg.get("fullgraph", False))
    dynamic = bool(compile_cfg.get("dynamic", False))
    inductor_cudagraphs = bool(
        compile_cfg.get("inductor_cudagraphs", True)
    )
    if backend == "inductor":
        import torch._inductor.config as inductor_config

        inductor_config.triton.cudagraphs = inductor_cudagraphs

    targets = _compile_targets(head, compile_cfg.get("modules", []))
    if not targets:
        raise RuntimeError("Compile was enabled but no modules were selected")
    for name, module in targets:
        module.compile(
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        _log(logger, f"[TrainingRuntime] compiled {name}")

    _log(
        logger,
        "[TrainingRuntime] selective compile configured with "
        f"backend={backend}, mode={mode}, fullgraph={fullgraph}, "
        f"dynamic={dynamic}, inductor_cudagraphs={inductor_cudagraphs}, "
        f"dynamo_cache_size_limit={dynamo_cache_size_limit}. "
        "Actual graph capture/fallback must be verified from the training log.",
    )
