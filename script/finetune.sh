torchrun --nproc_per_node 8 \
./finetune/run.py \
--output_dir ./bias_eliminator \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--train_data ./train_data/train_output.json \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--cutoff_len 4096 \
--logging_steps 1 \
--save_steps 3000 \
--save_total_limit 3 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed ./finetune/stage1.json \
--warmup_ratio 0.01 \
--fp16 \
--lora_alpha 32