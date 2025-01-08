import argparse
import json

import tiktoken
from tornado import concurrent
from tqdm import tqdm

from langchain_openai import ChatOpenAI

from prompts import *
from tools import *

def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--dev_path', type=str, default="dev.json")
    parser.add_argument('--data_path', type=str, default="preprocessed_data.json")
    parser.add_argument('--input_path', type=str, default="second_round.sql")
    parser.add_argument('--output_path', type=str, default="third_round.sql")
    parser.add_argument('--db_path', type=str, default="database")
    parser.add_argument('--retry_num', type=int, default=10)
    parser.add_argument('--process_num', type=int, default=1)
    parser.add_argument('--schema_path', type=str, default=None)
    parser.add_argument('--short_model_name', type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument('--long_model_name', type=str, default="gpt-3.5-turbo-16k-0613")

    opt = parser.parse_args()

    return opt


class ReflectTool:
    def __init__(self, short_model_name: str = "gpt-3.5-turbo-0613", long_model_name: str = "gpt-3.5-turbo-16k-0613"):
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.prompt_template_kg = reflect_prompt_kg
        self.prompt_template = reflect_prompt
        self.llm = ChatOpenAI(temperature=0, model_name=short_model_name, request_timeout=600, max_retries=3)
        self.llm_long = ChatOpenAI(temperature=0, model_name=long_model_name, request_timeout=600, max_retries=3)

    def run(self, question, schema, foreign_keys, previous_information, knowledge: None):
        if knowledge is not None:
            prompt = self.prompt_template_kg.format(question=question, schema=schema, foreign_keys=foreign_keys,
                                                    previous_information=previous_information, knowledge=knowledge).strip()
        else:
            prompt = self.prompt_template.format(question=question, schema=schema, foreign_keys=foreign_keys,
                                                  previous_information=previous_information).strip()
        prompt = '\n'.join([' '.join(e.split()) for e in prompt.split('\n')])

        if len(self.encoder.encode(prompt)) < 3800:
            reflect = self.llm.predict(prompt)
        else:
            reflect = self.llm_long.predict(prompt)

        return reflect


class CorrectTool():
    def __init__(self, short_model_name: str = "gpt-3.5-turbo-0613", long_model_name: str = "gpt-3.5-turbo-16k-0613"):
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.prompt_template_kg = correct_prompt_kg
        self.prompt_template = correct_prompt
        self.llm = ChatOpenAI(temperature=0, model_name=short_model_name, request_timeout=600, max_retries=3)
        self.llm_long = ChatOpenAI(temperature=0, model_name=long_model_name, request_timeout=600, max_retries=3)

    def run(self, question, schema, foreign_keys, previous_information, knowledge: None):
        if knowledge is not None:
            prompt = self.prompt_template_kg.format(question=question, schema=schema, foreign_keys=foreign_keys,
                                                    previous_information=previous_information,
                                                    knowledge=knowledge).strip()
        else:
            prompt = self.prompt_template.format(question=question, schema=schema, foreign_keys=foreign_keys,
                                                 previous_information=previous_information).strip()
        prompt = '\n'.join([' '.join(e.split()) for e in prompt.split('\n')])

        if len(self.encoder.encode(prompt)) < 3800:
            sql = 'SELECT ' + self.llm.predict(prompt)
        else:
            sql = 'SELECT ' + self.llm_long.predict(prompt)

        sql = sql.replace('```sql\n', '').replace('```\n', '').replace('\n```',
                                                                     '').replace('SELECTsql\n',
                                                                                 '').replace('```',
                                                                                             '')
        # print(sql)
        # print('===========================')

        sql = sql.replace("SELECT SELECT", "SELECT")
        sql = sql.replace('\n', ' ')
        sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
        last_sql = copy.deepcopy(sql)
        while last_sql != sql:
            last_sql = copy.deepcopy(sql)
            sql = sql.replace('  ', ' ')
        return sql


class SQLGenerateTool:
    def __init__(self, short_model_name: str = "gpt-3.5-turbo-0613", long_model_name: str = "gpt-3.5-turbo-16k-0613"):
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.prompt_template_kg = sql_middle_prompt_kg
        self.prompt_template = sql_middle_prompt
        self.llm = ChatOpenAI(temperature=0, model_name=short_model_name, request_timeout=600, max_retries=3)
        self.llm_long = ChatOpenAI(temperature=0.0, model_name=long_model_name, request_timeout=600, max_retries=3)

    def run(self, question, schema, foreign_keys, knowledge: None):
        if knowledge is not None:
            prompt = self.prompt_template_kg.format(question=question, schema=schema, foreign_keys=foreign_keys,
                                                    knowledge=knowledge).strip()
        else:
            prompt = self.prompt_template.format(question=question, schema=schema, foreign_keys=foreign_keys,
                                                 ).strip()
        prompt = '\n'.join([' '.join(e.split()) for e in prompt.split('\n')])
        if len(self.encoder.encode(prompt)) < 3800:
            sql = 'SELECT ' + self.llm.predict(prompt)
        else:
            sql = 'SELECT ' + self.llm_long.predict(prompt)

        global input_length, output_length
        input_length += len(self.encoder.encode(prompt))
        output_length += len(self.encoder.encode(sql))
        with open('third_round_input.txt', 'a') as f:
            f.write(str(input_length) + '\n')
        with open('third_round_output.txt', 'a') as f:
            f.write(str(output_length) + '\n')

        sql = sql.replace("SELECT SELECT", "SELECT")
        sql = sql.replace('\n', ' ')
        sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
        return sql


def correct_sql(idx, start, end, dev, data_all, sqls, output_path, db_path, MAX_RETRY_NUM, schema_path, short_model_name, long_model_name):
    reflector = ReflectTool(short_model_name, long_model_name)
    corrector = CorrectTool(short_model_name, long_model_name)
    generator = SQLGenerateTool(short_model_name, long_model_name)

    temp_output_path = get_output_name(output_path, idx)
    if os.path.exists(temp_output_path):
        with open(temp_output_path, 'r') as f:
            start = len(f.readlines()) + start
    else:
        start = start
    f_tmp = open(temp_output_path, 'a+')

    if schema_path is not None:
        all_schema = json.load(open(schema_path))
    else:
        all_schema = None

    error_num = 0
    error_not_solve = 0
    for i in tqdm(range(len(sqls))[start:end]):
        db_id = dev[i]['db_id']
        db_dir = f'{db_path}/{db_id}/{db_id}.sqlite'
        sql = sqls[i].strip()
        # print(db_dir, sql)
        result, flag = new_run_sql(db_dir, sql)
        # print(result)
        times = 0
        last_sqls = [sql]
        last_errors = [str(result)]

        question = dev[i]['question']
        knowledge = dev[i].get('evidence')
        if all_schema is None:
            foreign_keys = generate_foreign_key(data_all[i])
            schema = generate_schema_simple(data_all[i])
        else:
            foreign_keys = generate_foreign_key_by_tables(data_all[i], all_schema[i].keys())
            schema = generate_schema_by_dict_only(data_all[i], all_schema[i])

        previous_information = f"""### SQL: {last_sqls[-1]}\n### Error message: {last_errors[-1]}"""

        if flag is False:
            error_num += 1
        else:
            # print('execute successfully..')
            pass
        try:
            while flag is False:
                sql_reason = reflector.run(question, schema, foreign_keys, previous_information, knowledge)
                previous_information += f'\n### Error Reason: {sql_reason}'
                sql = corrector.run(question, schema, foreign_keys, previous_information, knowledge)
                result, flag = new_run_sql(db_dir, sql)
                previous_information += f'\n### new SQL: {sql}\n### Error message: {result}'
                last_sqls.append(sql)
                last_errors.append(str(result))
                times += 1
                # print(previous_information)
                if times >= MAX_RETRY_NUM:
                    error_not_solve += 1
                    break
        except:
            sql = sqls[i].strip()

        # print('one over..')
        f_tmp.write(sql + '\n')
        f_tmp.flush()
    f_tmp.close()
    # print(error_num)
    # print(error_not_solve)


def get_output_name(path, idx):
    paths = path.split('.')
    paths[-2] = paths[-2] + str(idx)
    return '.'.join(paths)


def correct_parallel(dev, data_all, sqls, output_path, db_path, MAX_RETRY_NUM, schema_path, short_model_name, long_model_name, PROCESS_NUM):
    contents = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(PROCESS_NUM):
            start = i * len(dev) // PROCESS_NUM
            end = min((i + 1) * len(dev) // PROCESS_NUM, len(dev))
            futures.append(executor.submit(correct_sql,
                                           i,
                                           start,
                                           end,
                                           dev,
                                           data_all,
                                           sqls,
                                           output_path,
                                           db_path,
                                           MAX_RETRY_NUM,
                                           schema_path,
                                           short_model_name,
                                           long_model_name))
        for future in concurrent.futures.as_completed(futures):
            content = future.result()
            contents.append(content)
    return contents

def main(opt):
    dev = json.load(open(opt.dev_path))
    data_all = json.load(open(opt.data_path))
    input_path = opt.input_path
    output_path = opt.output_path
    db_path = opt.db_path
    MAX_RETRY_NUM = opt.retry_num
    PROCESS_NUM = opt.process_num
    schema_path = opt.schema_path
    short_model_name = opt.short_model_name
    long_model_name = opt.long_model_name

    with open(input_path, 'r') as f:
        last_sqls = f.readlines()
    for i, e in enumerate(last_sqls):
        last_sqls[i] = e.strip()

    if os.path.exists(output_path):
        return None

    correct_parallel(dev, data_all, last_sqls, output_path, db_path, MAX_RETRY_NUM, schema_path, short_model_name, long_model_name, PROCESS_NUM)
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