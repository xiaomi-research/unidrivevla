# Training and Evaluation on Bench2Drive

The published LEAD final model has its complete training and evaluation chain
under `lead/`. This page documents that final chain. The repository does not
publish a separate generic Bench2Drive training recipe.

---

## LEAD closed-loop evaluation

Read [`lead/README.md`](../lead/README.md) for the Qwen initialization, the
three-epoch Stage 1 to nine-epoch motion Stage 3 chain, the `b2d_zoo`
environment, external asset requirements, and the expected metric sequences.

### Bench2Drive final evaluation

```bash
cd lead
export LEAD_CHECKPOINT=/path/to/iter_17037/global_step17037/mp_rank_00_model_states.pt
export LEAD_QWEN_BASE=/path/to/Qwen3-VL-2B-Instruct
export LEAD_PYTHON_BIN=/path/to/b2d_zoo/bin/python
export LEAD_CARLA_ROOT=/path/to/carla
export LEAD_ROUTE_SOURCE=/path/to/bench2drive220_routes

LEAD_STAGE3_DS87_EPOCH9_GPU_IDS=0,1,2,3,4,5,6,7 \
bash launch_latest_4hz_stage3_ds87_epoch9_comfort_full220.sh
```

This is the final Bench2Drive full-220 entrypoint. It validates the config,
checkpoint identity, motion anchor, CARLA installation, and route count before
starting route workers.

### Fail2Drive final evaluation

Fail2Drive is a separate dataset and CARLA installation. Use the release-local
single-lidar Generalization44 wrapper; it does not use the upstream
`official_tfv6_creeping` command:

```bash
cd lead
export LEAD_CHECKPOINT=/path/to/iter_17037/global_step17037/mp_rank_00_model_states.pt
export LEAD_QWEN_BASE=/path/to/Qwen3-VL-2B-Instruct
export LEAD_PYTHON_BIN=/path/to/b2d_zoo/bin/python
export LEAD_CARLA_ROOT=/path/to/fail2drive-carla
export LEAD_CARLA_PYTHON_BINDING=/path/to/fail2drive-carla/PythonAPI/carla/py38
export LEAD_ROUTE_SOURCE=/path/to/fail2drive_routes

LEAD_STAGE3_DS87_EPOCH9_LONGEST6_GPU_IDS=0,1,2,3,4,5,6,7 \
bash launch_latest_4hz_stage3_ds87_epoch9_fail2drive_seed0_single_lidar_creeping_visualization_generalization44_8gpu.sh
```

Bench2Drive and Fail2Drive use separate route sources and metrics, while the
tracking-ID repair is shared by the single published LEAD runtime.
