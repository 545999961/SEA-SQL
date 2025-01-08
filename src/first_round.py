import argparse
import json

import tiktoken
from tornado import concurrent
from tqdm import tqdm

from langchain_community.chat_models import ChatOpenAI

from prompts import *
from tools import *


def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--dev_path', type=str, default="dev.json")
    parser.add_argument('--short_model_name', type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument('--long_model_name', type=str, default="gpt-3.5-turbo-16k-0613")
    parser.add_argument('--data_path', type=str, default="../generate_datasets_bird/preprocessed_data.json")
    parser.add_argument('--output_path', type=str, default="../intermediate_datasets_bird/first_round.sql")
    parser.add_argument('--process_num', type=int, default=1)

    opt = parser.parse_args()

    return opt


class SQLGenerateTool:
    def __init__(self, short_model_name: str = "gpt-3.5-turbo-0613", long_model_name: str = "gpt-3.5-turbo-16k-0613"):
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.prompt_template_kg = sql_simple_prompt_kg
        self.prompt_template = sql_simple_prompt
        self.llm = ChatOpenAI(temperature=0, model_name=short_model_name, request_timeout=60, max_retries=10)
        self.llm_long = ChatOpenAI(temperature=0, model_name=long_model_name, request_timeout=60, max_retries=10)

    def run(self, question, schema, foreign_keys, knowledge: None):
        if knowledge is not None:
            prompt = self.prompt_template_kg.format(question=question, schema=schema,
                                                    foreign_keys=foreign_keys, knowledge=knowledge).strip()
        else:
            prompt = self.prompt_template.format(question=question, schema=schema,
                                                 foreign_keys=foreign_keys).strip()
        prompt = '\n'.join([' '.join(e.split()) for e in prompt.split('\n')])
        sql = ''
        while sql == '':
            try:
                if len(self.encoder.encode(prompt)) < 3800:
                    sql = 'SELECT  ' + self.llm.predict(prompt)
                else:
                    sql = 'SELECT  ' + self.llm_long.predict(prompt)
            except Exception as e:
                print(e)
        sql = sql.replace('```sql\n', '')
        sql = sql.replace(' ```\n', '')
        sql = sql.replace('```', '')
        sql = sql.replace("SELECT  SELECT", "SELECT ")
        sql = sql.replace("SELECT SELECT", "SELECT ")
        sql = sql.replace('\n', ' ')
        sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
        sql = sql.replace('SELECT  ', 'SELECT ')
        return sql


def get_output_name(path, idx):
    paths = path.split('.')
    paths[-2] = paths[-2] + str(idx)
    return '.'.join(paths)


def generate_sql(start, end, ids, dev, idx, output_path, data_all, short_model_name, long_model_name):
    sql_generator = SQLGenerateTool(short_model_name, long_model_name)

    temp_output_path = get_output_name(output_path, idx)
    if os.path.exists(temp_output_path):
        with open(temp_output_path, 'r') as f:
            start = len(f.readlines()) + start
    else:
        start = start

    f_tmp = open(temp_output_path, 'a+')


    for use_id in tqdm(ids[start:end]):
        question = dev[use_id]['question']
        knowledge = dev[use_id].get('evidence')
        foreign_keys = generate_foreign_key(data_all[use_id])
        schema = generate_schema(data_all[use_id])
        # print(schema)
        # input()
        # continue
        # foreign_keys = generate_foreign_key_by_tables(data_all[use_id], all_schema[use_id].keys())
        # schema = generate_schema_by_dict_only(data_all[use_id], all_schema[use_id])

        pre_sql = sql_generator.run(question, schema, foreign_keys, knowledge)
        f_tmp.write(pre_sql + '\n')
        f_tmp.flush()
    f_tmp.close()
    return 1


def generate_parallel(ids, dev, output_path, data_all, short_model_name, long_model_name, PROCESS_NUM):
    contents = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(PROCESS_NUM):
            futures.append(executor.submit(generate_sql,
                                           i * len(ids) // PROCESS_NUM,
                                           min((i + 1) * len(ids) // PROCESS_NUM, len(ids)),
                                           ids,
                                           dev,
                                           i,
                                           output_path,
                                           data_all,
                                           short_model_name,
                                           long_model_name))
        for future in concurrent.futures.as_completed(futures):
            content = future.result()
            contents.append(content)
    return contents


def main(opt):
    dev = json.load(open(opt.dev_path))
    data_all = json.load(open(opt.data_path))
    output_path = opt.output_path
    short_model_name = opt.short_model_name
    long_model_name = opt.long_model_name
    PROCESS_NUM = opt.process_num

    ids = range(len(dev))
    if os.path.exists(output_path):
        return None
    generate_parallel(ids, dev, output_path, data_all, short_model_name, long_model_name, PROCESS_NUM)
    f_sql = open(output_path, 'w')
    for i in range(PROCESS_NUM):
        with open(get_output_name(output_path, i), 'r') as f:
            data = f.readlines()
        os.remove(get_output_name(output_path, i))
        for pre_sql in data:
            f_sql.write(pre_sql.strip('\n') + '\n')
    f_sql.close()


if __name__ == "__main__":
    opt = parse_option()
    main(opt)