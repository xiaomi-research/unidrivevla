# nuScenes anchor inputs for PR-10 reproduction

The PR-10 public-main 2B Stage2/Stage3 configs at base `5fce121` name three existing arrays for detection, map, and
motion initialization.
Re-fitting them from the dataset is a new initialization, not a reproduction: the official KMeans calls leave
`random_state=None`.

| Original file under `nuScenes/data/kmeans/` | Imported native-profile name | Required shape |
|---|---|---:|
| `kmeans_det_900.npy` | `det_900.npy` | `(900, 11)` |
| `kmeans_map_100.npy` | `map_100.npy` | `(100, 20, 2)` |
| `kmeans_motion_6.npy` | `motion_6.npy` | `(10, 6, 12, 2)` |

The original arrays are not stored in this source checkout. The available baseline is the public main snapshot at
`setup/public-main-stage1.SDDVGe/source/nuScenes/data/kmeans`; its source and hashes are recorded in
`receipts/anchor_source.log`. This validates against the public main baseline only. It does not prove that a private
training run used byte-identical copies. Import those public baseline files into the profile's `anchors_dir` (by default
`$UNIDRIVE_NATIVE_ANCHORS_ROOT/native/nuscenes`). The importer validates names, shapes, floating dtype, and finite values,
copies the `.npy` files without converting their values, and records file and canonical array SHA-256 values:

```bash
python unidrivevla/tools/check_nuscenes_anchors.py \
  --source-dir /data/yongkangli/UniDriveVLA/setup/public-main-stage1.SDDVGe/source/nuScenes/data/kmeans \
  --target-dir /path/to/anchors/native/nuscenes

# Recheck the imported files against the saved manifest before training.
python unidrivevla/tools/check_nuscenes_anchors.py \
  --target-dir /path/to/anchors/native/nuscenes
```

For a reference independent of the import itself, first save a manifest from a trusted public main checkout, then pass it
to both import and later verification:

```bash
python unidrivevla/tools/check_nuscenes_anchors.py \
  --source-dir /data/yongkangli/UniDriveVLA/setup/public-main-stage1.SDDVGe/source/nuScenes/data/kmeans \
  --write-manifest /path/to/anchor-baselines/pr10-nuscenes.json

python unidrivevla/tools/check_nuscenes_anchors.py \
  --source-dir /data/yongkangli/UniDriveVLA/setup/public-main-stage1.SDDVGe/source/nuScenes/data/kmeans \
  --target-dir /path/to/anchors/native/nuscenes \
  --manifest /path/to/anchor-baselines/pr10-nuscenes.json

python unidrivevla/tools/check_nuscenes_anchors.py \
  --target-dir /path/to/anchors/native/nuscenes \
  --manifest /path/to/anchor-baselines/pr10-nuscenes.json
```

If an existing target file is byte-identical to the public source, the importer leaves it in place and fills any missing
files. A same-name file or manifest with conflicting content stops the import without changing that file. Use
`--replace-existing` only when you intentionally want to replace the whole anchor set. `array_sha256` hashes the array
shape, dtype, and C-order numeric bytes (independent of the `.npy` header);
`file_sha256` hashes the exact `.npy` file. A manifest is only as authoritative as the source directory used to create it.
This checkout has no anchor arrays, so the recorded public-main SHA values are not presented as private experiment hashes.
The target's auto-saved manifest checks later edits against the imported source; use a separately saved reference
manifest when you want to verify against the public source independently. Keep that reference manifest with the experiment
assets.

Use `python unidrivevla/tools/check_nuscenes_anchors.py --self-test` to exercise import, shape checks, hashing, and
corruption detection using small synthetic CPU arrays; this does not validate unavailable experiment assets.

## Re-fitting is a separate experiment

`unidrivevla/tools/fit_anchors.py` remains the fitting tool. With no seed, native LiDAR profiles retain the official
KMeans default (`random_state=None`); this preserves the old command behavior but does not make the result repeatable.
`--seed N` opts into seeded KMeans and seeded native Bench2Drive map-segment placement. The seed can make a fresh fit
repeatable within the same data, NumPy, and scikit-learn versions, but it does not recreate the PR-10 arrays. Existing
files still require `--replace-existing` to overwrite.

## Randomness and limits

The PR-10 training CLI defaults `--seed` to `0` and seeds the Python, NumPy, and PyTorch process RNGs. Its worker
initializer seeds NumPy and Python with `num_workers * rank + worker_id + seed`; PyTorch also initializes each worker's
torch RNG. The target Stage2/Stage3 recipes use eight workers per GPU and the native loader uses the same worker-seed
formula in their `pr10_numpy` mode.

Both official nuScenes geometry and photometric augmentation use `numpy.random`: the old pipeline imports `random` from
NumPy in `datasets/pipelines/augment.py`, and geometry uses NumPy in `nuscenes_3d_dataset.py`. Native Stage2 sets
`augment.geometry.rng_mode=pr10_numpy`; its geometry and `PhotoMetricDistortion` also use NumPy. The Stage3 recipe
inherits Stage2's setting. Generic recipes keep the old defaults (`num_workers=4`, Python-RNG geometry); the PR-10
overrides are limited to the dedicated Stage2/Stage3 recipe.

The selected nuScenes PR-10 path now aligns the RNG library, per-worker seed formula, worker count, and sequence-group
boundary checks. This is stronger than merely reusing seed `0`, but it is not a guarantee of identical full-run sample
ordering or weights. The PR-10 run used DeepSpeed ZeRO-1, while the native recipe uses the DDP trainer and its optimizer
path; rank scheduling, GPU kernels, and remaining nondeterministic operations still differ. The public-source receipt
does not document a private run's per-sample augmentation trace.

The public-main anchor arrays can be reused exactly by importing and SHA-checking those files. A newly seeded fit is
repeatable only under a fixed data/toolchain. Neither anchor hashes nor a shared seed guarantee identical training
weights or convergence: framework/distributed optimizer behavior, worker scheduling, GPU kernels, and nondeterministic
operations remain separate sources of variation.
