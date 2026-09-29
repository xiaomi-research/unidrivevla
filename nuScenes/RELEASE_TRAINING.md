# UniDriveVLA nuScenes release training

The six `unidrivevla_nusc_*.py` files in
`projects/configs/UniDriveVLA/` are the recipes captured from the released
2B/8B Stage 2 and Stage 3 checkpoints. They preserve the adapter ranks,
loss weights, epoch counts, batch sizes, ZeRO settings, motion anchor setup,
and evaluation switches used for those runs.

The original working tree contains many exploratory sweeps, ablations,
failed-run smoke scripts, and alternate CoC evaluators. They are intentionally
not part of this release. The shell files listed in the recipe table below are
the only canonical nuScenes training entry points; the shared
`train_stage2_coc_language_8gpu_workdirs2.sh` file is an implementation helper
used by the Stage 2 wrappers.

The repository does not contain model checkpoints, nuScenes sensor data, VAD
info files, or work directories. Set the paths below before launching a run:

```bash
export NUSCENES_ROOT=/path/to/unidrivevla-main/nuScenes
export NUSCENES_DATA_CWD=/path/to/nuScenes-data-working-copy
export VLM_PRETRAINED_PATH=/path/to/stage1_or_canbus_checkpoint
export OCCWORLD_VAE_PATH=/path/to/occvae_latest.pth
export DATA_INFOS_ROOT=$NUSCENES_DATA_CWD/data/infos
export NUSCENES_DATA_ROOT=/path/to/nuScenes/data/nuscenes
export NUSCENES_ANCHOR_ROOT=$NUSCENES_DATA_CWD/data/kmeans
```

`NUSCENES_ROOT` is the release code directory. `NUSCENES_DATA_CWD` is the
directory used as the process working directory so relative paths embedded in
nuScenes info files such as `./data/nuscenes/samples/...` resolve correctly.
When the data and the release are colocated, set both variables to the release
`nuScenes` directory. The launchers always invoke `tools/train.py`,
`tools/test.py`, and the prompt checks from this release directory, even when
`NUSCENES_DATA_CWD` points at an external data checkout.

`NUSCENES_ANCHOR_ROOT` must contain `kmeans_det_900.npy`,
`kmeans_map_100.npy`, and `kmeans_motion_6.npy`. These are external dataset
artifacts and are checked by every fixed training/evaluation launcher; no
checkpoint or sensor data is copied into this repository.

The CoC JSONL files used by Stage 2 and evaluation are included under
`data/nuscenes/` as gzip archives. Restore the original JSONL files before
training or evaluation:

```bash
python data/nuscenes/restore_alpamayo_jsonl.py
```

The restore script verifies both files against the original SHA-256 checksums.
The reconstructed files are ignored by Git because the training JSONL is
larger than GitHub's regular-file limit.

## Recipes

| Run | Config | Launcher |
| --- | --- | --- |
| 2B Stage 2 | `unidrivevla_nusc_2b_stage2.py` | `train_stage2_coc_weight01_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_8gpu_workdirs2.sh` |
| 2B Stage 2 with ego | `unidrivevla_nusc_2b_stage2_with_ego.py` | `train_stage2_coc_weight01_2b_canbus_sft3ep_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_8gpu_workdirs2.sh` |
| 2B Stage 3 motion | `unidrivevla_nusc_2b_stage3.py` | `train_unidrivevla_stage2_2b_coc_motion_15ep_8gpu.sh` |
| 8B Stage 2 | `unidrivevla_nusc_8b_stage2.py` | `train_stage2_large_8b_16gpu_2node.sh vision_beta0005_llm_beta0003` |
| 8B Stage 2 with ego | `unidrivevla_nusc_8b_stage2_with_ego.py` | `train_stage2_coc_weight01_8b_canbus_sft3ep_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_8gpu_workdirs2.sh` |
| 8B Stage 3 motion | `unidrivevla_nusc_8b_stage3.py` | `train_unidrivevla_stage3_8b_large_r128_top16_motion_15ep_16gpu_2node.sh` |

Stage 3 requires the preceding Stage 2 DeepSpeed checkpoint. Set
`STAGE1_CHECKPOINT` (the name is retained for compatibility with the original
launcher) to that checkpoint before starting Stage 3.

The output directory names captured by these recipes are:

| Recipe | Final output directory |
| --- | --- |
| 2B Stage 2 | `unidrivevla_stage2_2b_coc_weight01_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_r64_top16_bs4_accum4` |
| 2B Stage 3 | `unidrivevla_stage2_2b_coc_motion_anchor_only_15ep_from_iter17580_bs4_accum4_8gpu` |
| 2B Stage 2 with ego | `unidrivevla_stage2_2b_canbus_sft3ep_coc_weight01_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_r64_top16_bs4_accum4` |
| 8B Stage 2 | `unidrivevla_stage2_8b_nusc_large_stage1_coc_weight01_visual_wasp_beta0005_r128a256_top16_projector_lora_r128a256_llm_wasp_beta0003_r128a256_top16_2node16gpu_bs2_accum4_zero2_vgc` |
| 8B Stage 3 | `unidrivevla_stage3_8b_large_r128a256_top16_motion_15ep_from_stage2_epoch20_bs2_accum4_2node16gpu_zero2_vgc_noema_eager` |
| 8B Stage 2 with ego | `unidrivevla_stage2_8b_canbus_sft3ep_coc_weight01_visual_wasp_beta001_projector_lora_llm_wasp_beta0003_r64_top16_bs2_accum8_zero2_vgc` |

The Stage 3 recipes load the epoch-20 Stage 2 checkpoint at
`iter_17580/global_step4395/mp_rank_00_model_states.pt`. The 2B Stage 3
recipe runs for 13,185 iterations and the 8B Stage 3 recipe runs for 13,185
iterations with the two-node/16-GPU ZeRO-2 layout.

## Can-bus VLM SFT

Run `../pretrain_script/unidrivevla_nusc_8b_full_ego_canbus_pure_sft_3ep.sh`
for the 8B three-epoch full-parameter can-bus SFT. The resulting HF directory
is then supplied as `VLM_PRETRAINED_PATH` to the 8B ego Stage 2 launcher. The
2B and 8B ego Stage 2 launchers each accept their matching three-epoch
can-bus SFT checkpoint through `VLM_PRETRAINED_PATH`.

## Evaluation

Use `eval_nusc_8gpu.sh` for the six released recipes. Set `CFG` and `CKPT`
to the recipe being evaluated. The script enables `with_instance_id=True`;
the decoder always applies the temporal ID ownership transfer in evaluation,
so an ID cannot be emitted twice in one frame. There is no separate tracking
fix evaluation variant or tracking environment switch.
