tables="../DAMO-ConvAI/bird/llm/data/dev/dev_tables.json"
dev_path="../DAMO-ConvAI/bird/llm/data/dev/dev.json"
db_path="../DAMO-ConvAI/bird/llm/data/dev/dev_databases"
model_name="./bias_eliminator"
cache_dir="./data/LMs"
PROCESS_NUM=1
API_CALL_NUM=100
GPU_NUM=8

short_model_name='gpt-3.5-turbo-0613'
long_model_name='gpt-3.5-turbo-16k-0613'

first_output_path="./intermediate_datasets/first_round_test.sql"
second_output_path="./intermediate_datasets/second_round.sql"
third_output_path="./intermediate_datasets/third_round.sql"
processed_dataset_path="./generate_datasets/preprocessed_data.json"
final_output_path="./intermediate_datasets/predict_dev.json"

RETRY_NUM=10

directory="./intermediate_datasets"
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

directory="./generate_datasets"
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

current_time=$(date)
echo $current_time

echo "preprocessing..."
python preprocess/preprocessing.py \
    --mode "test" \
    --table_path $tables \
    --input_dataset_path $dev_path \
    --output_dataset_path $processed_dataset_path \
    --db_path $db_path \
    --target_type "sql" \
    --process_num $PROCESS_NUM

echo "first round..."
python src/first_round.py \
    --dev_path $dev_path \
    --data_path $processed_dataset_path \
    --output_path $first_output_path \
    --short_model_name $short_model_name \
    --long_model_name $long_model_name \
    --process_num $API_CALL_NUM

echo "second round..."
python src/second_round.py \
    --cache_dir $cache_dir \
    --dev_path $dev_path \
    --data_path $processed_dataset_path \
    --input_path $first_output_path \
    --output_path $second_output_path \
    --db_path $db_path \
    --num_gpus $GPU_NUM \
    --model_name %model_name

echo "third round..."
python src/third_round.py \
    --dev_path $dev_path \
    --data_path $processed_dataset_path \
    --input_path $second_output_path \
    --output_path $third_output_path \
    --db_path $db_path \
    --retry_num $RETRY_NUM \
    --short_model_name $short_model_name \
    --long_model_name $long_model_name \
    --process_num $API_CALL_NUM

echo "generate standard output..."
python src/append_db_id.py \
    --dev_path $dev_path \
    --input_path $first_output_path \
    --output_path $final_output_path

if [ ! -d "$third_output_path" ]; then
    sleep 5
fi

pkill -f third_round.pys

current_time=$(date)
echo $current_time