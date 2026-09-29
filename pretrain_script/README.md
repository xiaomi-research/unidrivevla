# NuScenes VQA and canbus pretraining

The selected 2B with-ego NuScenes result starts with a separate, full-model
Qwen3-VL SFT on the full-canbus planning QA JSONL for three epochs. That phase
is launched by:

```bash
bash pretrain_script/unidrivevla_nusc_2b_full_ego_canbus_pure_sft_3ep.sh
```

The script writes an HF-format model directory and records its path in
`latest_hf_model_path.txt`. The subsequent COC/WASP perception and planning
stage consumes that directory through `VLM_PRETRAINED_PATH`:

```bash
VLM_PRETRAINED_PATH=/path/to/pure_sft_checkpoint \
bash nuScenes/train_stage2_coc_weight01_2b_canbus_sft3ep_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_8gpu_workdirs2.sh
```

The release contains the launch logic, `system.txt`, and `tokens.txt`; the
base Qwen checkpoint, canbus JSONL files, output checkpoint, and ms-swift
installation remain external. Set `MODEL_PATH`, `EGO_CANBUS_TRAIN_JSONL`,
`EGO_CANBUS_VAL_JSONL`, `SAVE_DIR`, `PYTHON_BIN`, and optionally `SWIFT_ROOT`
for the local environment.
