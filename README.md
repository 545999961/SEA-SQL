## 📖 Introduction

This is the official repository for the paper ["SEA-SQL: Semantic-Enhanced Text-to-SQL with Adaptive Refinement"](https://arxiv.org/abs/2408.04919).

In this paper, we introduce Semantic-Enhanced Text-to-SQL with Adaptive Refinement (SEA-SQL), which includes **Adaptive Bias Elimination** and **Dynamic Execution Adjustment**, aims to improve performance while minimizing resource expenditure with zero-shot prompts

## ⚡ Environment

1. prepare python environment

```shell
conda create -n seasql python=3.10
conda activate seasql
pip install -r requirements.txt 
```

2. prepare API key

```
export OPENAI_API_BASE="..."
export OPENAI_API_KEY="..."
```

## 🔧 Data Preparation

You can get data from [BIRD](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) and [Spider](https://github.com/taoyds/spider).

The data should include the following files:
├─data # store datasets and databases
|  ├─dev
|    ├─dev_databases
|    ├─dev.json
|    ├─dev_tables.json
|    ├─dev_gold.sql
|  ├─train
|    ├─train_databases
|    ├─train.json
|    ├─train_tables.json
|    ├─train_gold.sql

## 🚀 Finetune
1. modify the data path in **generate_train_data.sh**

```shell
tables=".../data/train/train_tables.json"
dev_path=".../data/train/train.json"
db_root_path='.../data/train_databases/'
diff_json_path=".../data/train/train.json"
ground_truth_path='.../data/train/train_gold.sql'

short_model_name='gpt-3.5-turbo-0613'
long_model_name='gpt-3.5-turbo-16k-0613'
```

2. execute **generate_train_data.sh** to generate train data

```shell
bash script/generate_train_data.sh
```

3. finetune adaptive bias eliminator

```shell
bash script/finetune.sh
```

## 🫡 Run code

Our bias eliminator is available at [cfli/bias_eliminator](https://huggingface.co/cfli/bias_eliminator)

1. modify the data path in **script/run.sh**

```shell
tables=".../tables.json" # the path of tables.json
dev_path=".../dev.json" # the path of dev.json
db_path=".../database" # the database dir path
model_name="./bias_eliminator" # the path of bias eliminator
cache_dir="./LMs" # the dir to save the models (mistral-7B)
PROCESS_NUM=1 # the number of supported processes related to processing data sets
API_CALL_NUM=100 # the number of supported simultaneous API requests
GPU_NUM=1 # the number of GPUs to be used for inference

short_model_name='gpt-3.5-turbo-0613'
long_model_name='gpt-3.5-turbo-16k-0613'
```

2. execute **run.sh**

```shell
bash script/run.sh
```

## 📝 Evaluate

1. modify the data path in **script/evaluate.sh**

```shell
db_root_path='.../dev_databases/' # the database dir path
diff_json_path='.../dev/dev.json' # the path of dev.json
ground_truth_path='.../dev/' # the gold SQL dir path
```

2. execute **evaluate.sh**

```shell
bash script/evaluate.sh
```

## 💬Citation

If you find our work is helpful, please cite as:

```text
@misc{li2024seasqlsemanticenhancedtexttosqladaptive,
      title={SEA-SQL: Semantic-Enhanced Text-to-SQL with Adaptive Refinement}, 
      author={Chaofan Li and Yingxia Shao and Zheng Liu},
      year={2024},
      eprint={2408.04919},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2408.04919}, 
}
```

## 👍Contributing

We welcome contributions and suggestions!