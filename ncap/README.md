# UniDriveVLA NeuroNCAP evaluation

This directory contains the UniDriveVLA model HTTP adapter and the two
8-GPU launchers needed for a NeuroNCAP run:

1. The A server runs eight UniDriveVLA model replicas.
2. The B server runs the NeuroNCAP scenarios and starts one NeuRAD renderer
   per active task.

The release does not contain model checkpoints, base-model weights, nuScenes
data, NeuroNCAP data, NeuRAD checkpoints, logs, or result files. The model is
selected entirely through environment variables. `MODEL_KEY` is only an
output/log label and does not select a hard-coded checkpoint.

## A server: UniDriveVLA model replicas

Install the UniDriveVLA environment first. The NCAP adapter needs `peft` in
that environment:

```bash
pip install peft
```

Set the model and all external paths explicitly. `MODEL_ROOT` can point at a
full UniDriveVLA checkout, while `ADAPTER_ROOT` can remain the GitHub release
root so that the published `ncap/inference/server.py` is used.

```bash
cd /path/to/unidrivevla-main_github

MODEL_KEY=v003_l0003 \
MODEL_ROOT=/path/to/unidrivevla-main_github \
ADAPTER_ROOT=/path/to/unidrivevla-main_github/ncap \
CONFIG=/path/to/model_config.py \
CHECKPOINT=/path/to/checkpoint/mp_rank_00_model_states.pt \
VLM_PRETRAINED_PATH=/path/to/qwen3_vl_2b \
OCCWORLD_VAE_PATH=/path/to/occvae_latest.pth \
COC_JSONL_PATH=/path/to/alpamayo_coc_val_with_nav.jsonl \
NUSCENES_ROOT=/path/to/nuScenes \
DATA_INFOS_ROOT=/path/to/nuScenes/data/infos \
NUSCENES_DATA_ROOT=/path/to/nuScenes/data/nuscenes \
NUSCENES_ANCHOR_ROOT=/path/to/nuScenes/data/kmeans \
PYTHON_BIN=/path/to/model-env/bin/python \
HOST=0.0.0.0 \
bash ncap/run_ncap_model_server_2b_nomotion_8gpu.sh
```

The model server uses ports `9000` through `9007` and GPUs `0` through `7` by
default. Override `GPU_IDS`, `BASE_PORT`, `NUM_SERVERS`, `HOST`, and the paths
above when needed. To disable CoC generation for a model that does not use it,
set `COC_ENABLED=0`.

The script checks the config, checkpoint, base model, VAE, and the external
nuScenes info/data/anchor directories before starting. It does not copy any
of them. `NUSCENES_ROOT` and the three data variables may point outside the
model checkout; this is required when using the source-only GitHub release.

## B server: full NeuRAD evaluation

The B server needs the external NeuroNCAP and NeuRAD checkouts, including
their downloaded data and NeuRAD scene checkpoints. The NeuroNCAP Python
environment must provide the packages from its `pyproject.toml`; the NeuRAD
environment must provide `nerfstudio`, `viser`, `jaxtyping`, `tyro`, `torch`,
`fastapi`, and `uvicorn`.

```bash
cd /path/to/neuro-ncap

MODEL_KEY=v003_l0003 \
NCAP_ROOT=/path/to/neuro-ncap \
NEURAD_ROOT=/path/to/neurad-studio \
NUSCENES_ROOT=/path/to/nuScenes/data/nuscenes \
MODEL_HOST=10.96.55.164 \
PYTHON_BIN=/path/to/neuro-ncap-env/bin/python \
NEURAD_PYTHON_BIN=/path/to/neurad-env/bin/python \
RESUME=1 \
bash /path/to/unidrivevla-main_github/ncap/run_unidrivevla_2b_nomotion_full8_neurad.sh
```

`NUSCENES_ROOT` must contain `v1.0-trainval`. The runner uses all 20 standard
stationary, frontal, and side scenario sequences and 100 runs per sequence by
default. Use `RUNS`, `SELECTED_TASK_RUNS`, `START_TASK_INDEX`,
`MODEL_PORTS`, `RENDERER_GPUS`, and `RENDERER_BASE_PORT` to select a smaller
or different run. Results and logs are written under `NCAP_ROOT` unless
`RESULT_ROOT` and `LOG_ROOT` are supplied.

The model server and NeuroNCAP runner are separate processes and machines.
Start the A-server launcher first, verify that the eight `/alive` endpoints
respond, then start the B-server launcher.
