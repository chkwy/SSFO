echo "-------------------- Starting evaluation --------------------"
cd eval
dfo_model=${1:-/data/DFO/saves/test_qwen_model_tang}
dfo_name=$(basename "$dfo_model")
model_path=${2:-/home/YiWang/DFO-script/save_models/context-dpo-qwen2.5-7b-instruct}
model_name=$(basename "$model_path")
datasets=( "nqopen" "nqswap" "memo-trap" )
base_model_name=$model_name
base_dfo_model_name=$dfo_name
for dataset in "${datasets[@]}"; do
    echo "Evaluating $base_model_name on dataset $dataset"
    python main.py --model-id "$model_path" --eval-dataset "$dataset" --gpu_num 4 >> ../logs/${model_name}_run.log 2>&1

    echo "Evaluating $base_dfo_model_name on dataset $dataset"
    python main.py --model-id "$dfo_model" --eval-dataset "$dataset" --gpu_num 4 >> ../logs/${model_name}_run.log 2>&1
done

datasets=( "nqopen" "nqswap" "memo-trap" )
echo "-------------------- Evaluation results --------------------"

printf "%-20s | %-25s | %-25s\n" "Dataset" "Base model Subspan_EM" "DFO model Subspan_EM"
printf "%-20s-+-%-25s-+-%-25s\n" "$(printf '%.0s-' {1..20})" "$(printf '%.0s-' {1..25})" "$(printf '%.0s-' {1..25})"

for dataset in "${datasets[@]}"; do
    base_metric=$(cat outputs/${dataset}_${model_name}_metrics_final.json | \
                  tr ',' '\n' | grep '"sub_Subspan_EM"' | cut -d':' -f2 | tr -d ' }' | \
                  awk '{printf "%.2f%%", $1*100}')

    dfo_metric=$(cat outputs/${dataset}_${dfo_name}_metrics_final.json | \
                 tr ',' '\n' | grep '"sub_Subspan_EM"' | cut -d':' -f2 | tr -d ' }' | \
                 awk '{printf "%.2f%%", $1*100}')

    printf "%-20s | %-25s | %-25s\n" "$dataset" "$base_metric" "$dfo_metric"
done
