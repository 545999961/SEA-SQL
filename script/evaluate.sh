db_root_path='../DAMO-ConvAI/bird/llm/data/dev/dev_databases/'
diff_json_path='../DAMO-ConvAI/bird/llm/data/dev/dev.json'
ground_truth_path='../DAMO-ConvAI/bird/llm/data/dev/'
predicted_sql_path_kg='./intermediate_datasets/'
data_mode='dev'
num_cpus=100
meta_time_out=30.0
mode_gt='gt'
mode_predict='gpt'

echo '''starting to compare with knowledge for ex'''
python3 -u evaluation/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

echo '''starting to compare with knowledge for ves'''
python3 -u evaluation/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}