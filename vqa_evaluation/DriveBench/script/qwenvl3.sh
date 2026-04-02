GPU=$1
MODEL=${MODEL:-"/path/to/UniDriveVLA_nuScenes_hf"}
DATA_ROOT=${DATA_ROOT:-"."}

# Define corresponding output names and corruption values in order.
corruptions=(
    ""              # clean
)

outputs=(
    "clean"
)

# Loop over the arrays and run the commands.
for i in "${!outputs[@]}"; do
    python3 inference/qwenvl3_vllm.py \
        --model "$MODEL" \
        --data data/drivebench-test-final.json \
        --output "res/qwenvl3/${outputs[i]}" \
        --system_prompt prompt.txt \
        --num_processes "${GPU}" \
        --corruption "${corruptions[i]}" \
        --data_root "$DATA_ROOT"
done
