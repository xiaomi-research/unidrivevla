"""PR 10 release numbers in config/stage and config/profile/lead.yaml.

Hydra is not required: stage files that list `defaults:` are deep-merged in that
order, then the file itself overrides. This is the same overlay the stage group uses.
"""
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "config" / "stage"
LLM = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VIS = ["qkv", "proj", "linear_fc1", "linear_fc2"]
PROJ = r"^(merger|deepstack_merger_list\.\d+)\.linear_fc[12]$"


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def merge(base, over):
    out = dict(base)
    for key, value in over.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def stage(name, seen=None):
    seen = seen or set()
    if name in seen:
        raise SystemExit(f"stage defaults loop at {name}")
    seen.add(name)
    raw = load(STAGE / f"{name}.yaml")
    acc = {}
    for item in raw.get("defaults") or []:
        if isinstance(item, str):
            acc = merge(acc, stage(item, seen))
    return merge(acc, raw)


def expect(cfg, path, value, label):
    got = cfg
    for key in path.split("."):
        got = got[key]
    if isinstance(value, float):
        if abs(float(got) - value) > 1e-12:
            raise SystemExit(f"{label}: {path} is {got}, expected {value}")
    elif list(got) != list(value) if isinstance(value, list) else got != value:
        raise SystemExit(f"{label}: {path} is {got}, expected {value}")


def wasp(rank, alpha, beta, layers, weight=None):
    cfg = {"type": "wasp", "rank": rank, "alpha": alpha, "top_k": 16, "beta": beta,
           "dropout": 0.0, "eps": 1e-6, "expected_num_layers": layers, "target_modules": VIS}
    if weight is not None:
        cfg["wasp_loss_weight"] = weight
    return cfg


def llm(rank, alpha, beta, layers, weight=None):
    cfg = {"type": "wasp", "rank": rank, "alpha": alpha, "top_k": 16, "beta": beta,
           "dropout": 0.0, "eps": 1e-6, "expected_num_layers": layers, "target_modules": LLM}
    if weight is not None:
        cfg["wasp_loss_weight"] = weight
    return cfg


def check_nusc(name, **kw):
    cfg = stage(name)
    expect(cfg, "train.max_epochs", kw["epochs"], name)
    expect(cfg, "train.global_batch", 128, name)
    expect(cfg, "train.grad_clip", 0.0, name)
    expect(cfg, "dataloader.params.batch_size", kw["batch"], name)
    expect(cfg, "agent.lr", kw["lr"], name)
    expect(cfg, "agent.weight_decay", 1e-7, name)
    expect(cfg, "agent.vlm_lr_mult", 0.5, name)
    expect(cfg, "agent.warmup_start", 0.3333333333333333, name)
    expect(cfg, "agent.min_lr_ratio", 0.001, name)
    expect(cfg, "agent.warmup_steps", kw["warmup"], name)
    expect(cfg, "agent.vlm_variant", kw["variant"], name)
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
    common = dict(epochs=20, lr=2e-4, warmup=125, motion=False, ego=False, proj_r=64, proj_alpha=128,
                  visual=wasp(64, 128, 0.01, 96), llm=llm(64, 128, 0.003, 196), batch=4, variant="2b")
    check_nusc("nusc_2b_stage2", **common)
    check_nusc("nusc_2b_stage2_ego", **{**common, "ego": True})
    check_nusc("nusc_2b_stage3", **{**common, "epochs": 15, "lr": 1e-4, "motion": True})
    check_nusc("nusc_8b_stage2", **{**common, "variant": "8b", "batch": 2, "proj_r": 128, "proj_alpha": 256,
                                    "visual": wasp(128, 256, 0.005, 108), "llm": llm(128, 256, 0.003, 252)})
    check_nusc("nusc_8b_stage2_ego", **{**common, "variant": "8b", "batch": 2, "warmup": 62.5, "ego": True,
                                        "visual": wasp(64, 128, 0.01, 108), "llm": llm(64, 128, 0.003, 252)})
    check_nusc("nusc_8b_stage3", **{**common, "variant": "8b", "batch": 2, "epochs": 15, "lr": 1e-4, "motion": True,
                                    "proj_r": 128, "proj_alpha": 256,
                                    "visual": wasp(128, 256, 0.0, 108, 0.0), "llm": llm(128, 256, 0.0, 252, 0.0)})
    sft = stage("canbus_sft")
    expect(sft, "train.max_epochs", 3, "canbus_sft")
    expect(sft, "agent.lr", 4e-5, "canbus_sft")
    expect(sft, "agent.vlm_lr_mult", 1.0, "canbus_sft")
    expect(sft, "train.global_batch", 128, "canbus_sft")
    expect(sft, "dataloader.params.batch_size", 1, "canbus_sft")
    expect(sft, "vqa.max_pixels", 522240, "canbus_sft")
    lead = merge(load(ROOT / "config" / "data" / "lead.yaml"), stage("lead_stage1"))
    motion = merge(lead, stage("lead_motion"))
    profile = load(ROOT / "config" / "profile" / "lead.yaml")
    expect(lead, "agent.lr", 2e-4, "lead_stage1")
    expect(lead, "train.max_epochs", 3, "lead_stage1")
    expect(lead, "train.global_batch", 512, "lead_stage1")
    expect(lead, "dataloader.params.batch_size", 16, "lead_stage1")
    expect(lead, "agent.with_motion", False, "lead_stage1")
    expect(motion, "agent.lr", 1e-4, "lead_motion")
    expect(motion, "train.max_epochs", 9, "lead_motion")
    expect(motion, "agent.with_motion", True, "lead_motion")
    expect(motion, "train.global_batch", 512, "lead_motion")
    if profile["horizon"] != 8 or abs(float(profile["pose_interval"]) - 0.25) > 1e-12:
        raise SystemExit(f"LEAD profile horizon/interval is {profile['horizon']}, {profile['pose_interval']}")
    print("ok canbus_sft lead_stage1 lead_motion")


if __name__ == "__main__":
    main()
