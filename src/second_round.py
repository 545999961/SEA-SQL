import argparse
import json
import multiprocessing
import os
import time

import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools import *
from prompts import *

def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--model_name', type=str, default="./bias_eliminator")
    parser.add_argument('--cache_dir', type=str, default="LMs")

    parser.add_argument('--dev_path', type=str, default="dev.json")
    parser.add_argument('--data_path', type=str, default="../generate_datasets_bird/preprocessed_data.json")
    parser.add_argument('--input_path', type=str, default="../intermediate_datasets_bird/first_round.sql")
    parser.add_argument('--db_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default="../intermediate_datasets_bird/second_round.sql")
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--schema_path', type=str, default=None)

    opt = parser.parse_args()

    return opt


def generate_sql(idx, opt):
    model_name = opt.model_name
    cache_dir = opt.cache_dir
    sql_path = opt.input_path
    dev_path = opt.dev_path
    data_path = opt.data_path
    output_path = opt.output_path
    db_path = opt.db_path
    num_gpus = opt.num_gpus
    schema_path = opt.schema_path

    ### tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ### model
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16)

    model.eval()
    # time.sleep(60)
    # model.half()
    model = model.to(f'cuda:{idx}')
    ### load sql
    sqls = []
    with open(sql_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sqls.append(line.strip('\n'))
    ### load dev
    dev = json.load(open(dev_path))
    ### load preprocessed data
    data_all = json.load(open(data_path))

    temp_output_path = get_output_name(output_path, idx)

    start = idx * len(sqls) // num_gpus
    end = min((idx + 1) * len(sqls) // num_gpus, len(sqls))

    if os.path.exists(temp_output_path):
        with open(temp_output_path, 'r') as f:
            start = len(f.readlines()) + start
    else:
        start = start

    f = open(temp_output_path, 'w')

    if schema_path is not None:
        all_schema = json.load(open(schema_path))
    else:
        all_schema = None

    excepts = 0
    for i in trange(start, end):
        db_id = dev[i]['db_id']
        db_dir = f'{db_path}/{db_id}/{db_id}.sqlite'
        question = dev[i]['question']
        knowledge = dev[i].get('evidence')
        if all_schema is None:
            foreign_keys = generate_foreign_key(data_all[i])
            schema = generate_schema(data_all[i])
        else:
            foreign_keys = generate_foreign_key_by_tables(data_all[i], all_schema[i].keys())
            schema = generate_schema_by_dict_only(data_all[i], all_schema[i])

        result, flag = new_run_sql(db_dir, sqls[i])
        if knowledge is not None:
            input_prompt = bias_eliminator_prompt.format(schema=schema,
                                                         foreign_keys=foreign_keys,
                                                         question=question,
                                                         SQL=sqls[i],
                                                         result=result)
        else:
            input_prompt = bias_eliminator_prompt_with_knowledge.format(schema=schema,
                                                                        knowledge=knowledge,
                                                                        foreign_keys=foreign_keys,
                                                                        question=question,
                                                                        SQL=sqls[i],
                                                                        result=result)
        inputs = tokenizer(input_prompt, return_tensors='pt').to(f'cuda:{idx}')
        if len(inputs['input_ids'][0]) > 8000:
            excepts += 1
            new_sql = sqls[i]
        else:
            try:
                generate_ids = model.generate(inputs.input_ids, do_sample=False, max_length=8192)
                new_sql = tokenizer.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0].strip('\n').strip()
            except Exception as e:
                print(e)
                new_sql = sqls[i]

        f.write(new_sql + '\n')
        f.flush()
    print(excepts)
    f.close()

def get_output_name(path, idx):
    paths = path.split('.')
    paths[-2] = paths[-2] + str(idx)
    return '.'.join(paths)

def merge(num_gpus, output_path):
    all_sqls = []
    for i in range(num_gpus):
        with open(get_output_name(output_path, i)) as f:
            lines = f.readlines()
            for line in lines:
                all_sqls.append(line.strip('\n'))
        os.remove(get_output_name(output_path, i))
    with open(output_path, 'w') as f:
        for sql in all_sqls:
            f.write(sql + '\n')

if __name__ == "__main__":
    opt = parse_option()
    if os.path.exists(opt.output_path):
        sys.exit()
    processes = []
    multiprocessing.set_start_method('spawn')
    for i in range(opt.num_gpus):
        process = multiprocessing.Process(target=generate_sql, args=(i,opt, ))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    merge(opt.num_gpus, opt.output_path)