"""Check the release recipes after composing the same root Hydra config as train.py."""
from pathlib import Path

from hydra import compose as hydra_compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
NUSC_STATUS = ["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z", "vel_x", "vel_y", "vel_z", "steer"]
LLM = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VIS = ["qkv", "proj", "linear_fc1", "linear_fc2"]
PROJ = r"^(merger|deepstack_merger_list\.\d+)\.linear_fc[12]$"


def recipe(data: str, stage: str, *overrides: str):
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = hydra_compose("train", overrides=[f"data={data}", f"stage={stage}", *overrides])
    return OmegaConf.to_container(cfg, resolve=False)


def expect(cfg, path, value, label):
    got = cfg
    for key in path.split("."):
        got = got[key]
    if isinstance(value, float):
        if abs(float(got) - value) > 1e-12:
            raise SystemExit(f"{label}: {path} is {got}, expected {value}")
    elif list(got) != list(value) if isinstance(value, list) else got != value:
        raise SystemExit(f"{label}: {path} is {got}, expected {value}")


def respect(beta, layers, targets):
    return {"type": "wasp", "rank": 64, "alpha": 128, "top_k": 16, "beta": beta,
            "dropout": 0.0, "eps": 1e-6, "expected_num_layers": layers, "target_modules": targets}


def check_nusc(name, **kw):
    cfg = recipe("nuscenes", name)
    expect(cfg, "train.max_epochs", kw["epochs"], name)
    expect(cfg, "train.global_batch", 128, name)
    expect(cfg, "train.grad_clip", kw.get("grad_clip", 0.0), name)
    expect(cfg, "profile.official_command_text", kw.get("official_command_text", True), name)
    expect(cfg, "profile.status_layout", NUSC_STATUS, name)
    expect(cfg, "dataloader.params.batch_size", kw["batch"], name)
    expect(cfg, "agent.lr", kw["lr"], name)
    expect(cfg, "agent.weight_decay", 1e-7, name)
    expect(cfg, "agent.vlm_lr_mult", 0.5, name)
    expect(cfg, "agent.warmup_start", 0.3333333333333333, name)
    expect(cfg, "agent.min_lr_ratio", 0.001, name)
    expect(cfg, "agent.warmup_steps", kw["warmup"], name)
    expect(cfg, "agent.vlm_variant", kw["variant"], name)
    expect(cfg, "agent.train_vlm", True, name)
    expect(cfg, "agent.with_motion", kw["motion"], name)
    expect(cfg, "agent.coc_language_loss_weight", 0.1, name)
    expect(cfg, "agent.coc_planning_loss_weight", 0.0, name)
    expect(cfg, "agent.coc_language_prompt", True, name)
    expect(cfg, "agent.canbus_sft_prompt", kw["ego"], name)
    expect(cfg, "agent.visual_wasp", kw["visual"], name)
    expect(cfg, "agent.llm_wasp", kw["llm"], name)
    expect(cfg, "agent.visual_lora.r", kw["proj_r"], name)
    expect(cfg, "agent.visual_lora.lora_alpha", kw["proj_alpha"], name)
    expect(cfg, "agent.visual_lora.target_modules", PROJ, name)
    expect(cfg, "vqa.enabled", False, name)
    if "COC_JSONL_PATH" not in str(cfg["agent"]["coc_jsonl_path"]):
        raise SystemExit(f"{name}: CoC jsonl path is not COC_JSONL_PATH")
    print("ok", name)


def main():
    # The original main recipes keep the root defaults and data-specific epoch counts.
    main_nusc = recipe("nuscenes", "stage1")
    expect(main_nusc, "train.max_epochs", 30, "main_nuscenes_stage1")
    expect(main_nusc, "train.global_batch", 128, "main_nuscenes_stage1")
    expect(main_nusc, "train.grad_clip", 1.0, "main_nuscenes_stage1")
    expect(main_nusc, "dataloader.params.batch_size", 4, "main_nuscenes_stage1")
    main_b2d = recipe("b2d", "stage1")
    expect(main_b2d, "train.max_epochs", 10, "main_b2d_stage1")
    expect(main_b2d, "train.global_batch", 128, "main_b2d_stage1")
    expect(main_b2d, "train.grad_clip", 1.0, "main_b2d_stage1")
    expect(main_b2d, "dataloader.params.batch_size", 4, "main_b2d_stage1")
    print("ok main_nuscenes_stage1 main_b2d_stage1")

    check_nusc("nusc_2b_stage2", epochs=20, lr=2e-4, warmup=125, motion=False, ego=False, proj_r=64,
               proj_alpha=128, visual=respect(0.01, 96, VIS), llm=respect(0.003, 196, LLM), batch=4,
               variant="2b", grad_clip=1.0, official_command_text=False)

    # Stage 3 is an explicit continuation of the released Stage 2 recipe.  Its VLM/RESPECT/projector
    # module layout must stay identical so the old DeepSpeed planning_head.* tensors can migrate by name.
    stage2 = recipe("nuscenes", "nusc_2b_stage2")
    stage3 = recipe("nuscenes", "nusc_2b_stage3")
    expect(stage2, "agent.checkpoint_frame", "lidar", "nusc_2b_stage2")
    expect(stage2, "augment.geometry.rng_mode", "pr10_numpy", "nusc_2b_stage2")
    expect(stage2, "dataloader.params.num_workers", 8, "nusc_2b_stage2")
    expect(stage2, "dataloader.params.pin_memory", False, "nusc_2b_stage2")
    expect(stage2, "dataloader.params.persistent_workers", False, "nusc_2b_stage2")
    expect(stage2, "evaluator.name", "nuscenes_pr10", "nusc_2b_stage2")
    expect(stage2, "evaluator.every_n_epochs", 0, "nusc_2b_stage2")
    expect(stage2, "evaluator.on_end", True, "nusc_2b_stage2")
    expect(stage3, "agent.train_vlm", False, "nusc_2b_stage3")
    expect(stage3, "agent.with_motion", True, "nusc_2b_stage3")
    expect(stage3, "agent.checkpoint_path", "${oc.env:STAGE2_CHECKPOINT}", "nusc_2b_stage3")
    expect(stage3, "agent.checkpoint_frame", "lidar", "nusc_2b_stage3")
    expect(stage3, "agent.checkpoint_transition", "nusc_2b_stage2_to_stage3", "nusc_2b_stage3")
    expect(stage3, "agent.lr", 1.0e-4, "nusc_2b_stage3")
    expect(stage3, "agent.weight_decay", 1.0e-7, "nusc_2b_stage3")
    expect(stage3, "agent.warmup_steps", 125, "nusc_2b_stage3")
    expect(stage3, "agent.warmup_start", 0.3333333333333333, "nusc_2b_stage3")
    expect(stage3, "agent.min_lr_ratio", 0.001, "nusc_2b_stage3")
    expect(stage3, "agent.vlm_lr_mult", 0.5, "nusc_2b_stage3")
    expect(stage3, "agent.gradient_checkpointing", True, "nusc_2b_stage3")
    expect(stage3, "train.max_epochs", 15, "nusc_2b_stage3")
    expect(stage3, "train.global_batch", 128, "nusc_2b_stage3")
    expect(stage3, "train.grad_clip", 1.0, "nusc_2b_stage3")
    expect(stage3, "dataloader.params.batch_size", 4, "nusc_2b_stage3")
    expect(stage3, "dataloader.params.num_workers", 8, "nusc_2b_stage3")
    expect(stage3, "dataloader.params.pin_memory", False, "nusc_2b_stage3")
    expect(stage3, "dataloader.params.persistent_workers", False, "nusc_2b_stage3")
    expect(stage3, "augment.geometry.rng_mode", "pr10_numpy", "nusc_2b_stage3")
    expect(stage3, "profile.prefix_max_len", 3840, "nusc_2b_stage3")
    expect(stage3, "profile.official_attention_mask", False, "nusc_2b_stage3")
    expect(stage3, "profile.official_command_text", False, "nusc_2b_stage3")
    expect(stage3, "profile.lr_microstep", True, "nusc_2b_stage3")
    expect(stage3, "profile.motion_anchor_encoder", "MotionAnchorSineEncoder", "nusc_2b_stage3")
    expect(stage3, "agent.attn_implementation", "sdpa", "nusc_2b_stage3")
    expect(stage3, "profile.hf_attn_impl", "flash_attention_2", "nusc_2b_stage3")
    expect(stage3, "agent.coc_language_prompt", False, "nusc_2b_stage3")
    expect(stage3, "agent.coc_language_loss_weight", 0.0, "nusc_2b_stage3")
    expect(stage3, "agent.coc_jsonl_path", "", "nusc_2b_stage3")
    expect(stage3, "agent.coc_planning_loss_weight", 0.0, "nusc_2b_stage3")
    expect(stage3, "vqa.enabled", False, "nusc_2b_stage3")
    expect(stage3, "evaluator.name", "nuscenes_pr10_motion", "nusc_2b_stage3")
    expect(stage3, "evaluator.every_n_epochs", 0, "nusc_2b_stage3")
    expect(stage3, "evaluator.on_end", True, "nusc_2b_stage3")
    for path in ("agent.visual_wasp", "agent.visual_lora", "agent.llm_wasp"):
        expect(stage3, path, stage2["agent"][path.split(".")[1]], "nusc_2b_stage3")
    print("ok nusc_2b_stage3 inherits adapters and Stage-2 schedule")

    lead = recipe("lead", "lead_stage1")
    profile = lead["profile"]
    expect(lead, "agent.lr", 2e-4, "lead_stage1")
    expect(lead, "train.max_epochs", 3, "lead_stage1")
    expect(lead, "train.global_batch", 512, "lead_stage1")
    expect(lead, "dataloader.params.batch_size", 16, "lead_stage1")
    expect(lead, "agent.with_motion", False, "lead_stage1")
    expect(lead, "agent.train_vlm", True, "lead_stage1")
    if profile["horizon"] != 8 or abs(float(profile["pose_interval"]) - 0.25) > 1e-12:
        raise SystemExit(f"LEAD profile horizon/interval is {profile['horizon']}, {profile['pose_interval']}")
    print("ok lead_stage1")

    overridden = recipe("nuscenes", "nusc_2b_stage2", "train.global_batch=64", "train.grad_clip=0.5",
                        "dataloader.params.batch_size=1")
    expect(overridden, "train.global_batch", 64, "cli_overrides")
    expect(overridden, "train.grad_clip", 0.5, "cli_overrides")
    expect(overridden, "dataloader.params.batch_size", 1, "cli_overrides")
    alternate_group = recipe("nuscenes", "stage1", "vqa=alternate")
    expect(alternate_group, "vqa.enabled", True, "cli_vqa_group_override")
    expect(alternate_group, "vqa.mode", "alternate", "cli_vqa_group_override")
    print("ok cli_overrides")


if __name__ == "__main__":
    main()
