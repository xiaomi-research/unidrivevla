export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0

python tools/data_converter/vad_nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag vad_nuscenes \
    --version v1.0
