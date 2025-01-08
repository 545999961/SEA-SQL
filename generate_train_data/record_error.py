import os
import sys
import json
import argparse
import sqlite3
import multiprocessing as mp

import tiktoken
from func_timeout import func_timeout, FunctionTimedOut
from langchain_openai import ChatOpenAI

from tools import *
from prompts import *

class QAGenerateTool:
    def __init__(self):
        self.encoder = tiktoken.encoding_for_model("text-davinci-003")
        self.prompt_template_kg = generate_question_answer_with_knowledge
        self.prompt_template = generate_question_answer_without_knowledge
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613", request_timeout=60, max_retries=1)
        self.llm_long = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613", request_timeout=60, max_retries=1)

    def run(self, question, schema, foreign_keys, sql, knowledge: None):
        if knowledge is not None:
            prompt = self.prompt_template_kg.format(question=question,
                                                    schema=schema,
                                                    foreign_keys=foreign_keys,
                                                    knowledge=knowledge,
                                                    sql=sql).strip()
        else:
            prompt = self.prompt_template.format(question=question,
                                                 schema=schema,
                                                 foreign_keys=foreign_keys,
                                                 sql=sql).strip()
        prompt = '\n'.join([' '.join(e.split()) for e in prompt.split('\n')])
        try:
            if len(self.encoder.encode(prompt)) < 3800:
                result = self.llm.predict(prompt)
            else:
                result = self.llm_long.predict(prompt)
        except:
            if len(self.encoder.encode(prompt)) < 3800:
                self.llm.temperature = 0.5
                result = self.llm.predict(prompt)
                self.llm.temperature = 0
            else:
                self.llm_long.temperature = 0.5
                result = self.llm_long.predict(prompt)
                self.llm_long.temperature = 0
        if knowledge is None:
            new_question = result[result.find('New Question: ') + len('New Question: '): result.find('New Gold SQL:')].strip('\n').strip()
            new_sql = result[result.find('New Gold SQL: ') + len('New Gold SQL: '): ].strip('\n').strip()
            new_knowledge = None
        else:
            new_question = result[result.find('New Question: ') + len('New Question: '): result.find('New External Knowledge:')].strip('\n').strip()
            new_knowledge = result[result.find('New External Knowledge: ') + len('New External Knowledge: '): result.find('New Gold SQL:')].strip('\n').strip()
            new_sql = result[result.find('New Gold SQL: ') + len('New Gold SQL: '):].strip('\n').strip()
        # print(new_knowledge)
        # print(new_question)
        # print(new_sql)
        return new_question, new_sql, new_knowledge

class SQLGenerateTool:
    def __init__(self):
        self.encoder = tiktoken.encoding_for_model("text-davinci-003")
        self.prompt_template_kg = sql_simple_prompt_kg
        self.prompt_template = sql_simple_prompt
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613", request_timeout=60, max_retries=1)
        self.llm_long = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613", request_timeout=60, max_retries=1)

    def run(self, question, schema, foreign_keys, knowledge: None):
        if knowledge is not None:
            prompt = self.prompt_template_kg.format(question=question, schema=schema,
                                                    foreign_keys=foreign_keys, knowledge=knowledge).strip()
        else:
            prompt = self.prompt_template.format(question=question, schema=schema,
                                                 foreign_keys=foreign_keys).strip()
        prompt = '\n'.join([' '.join(e.split()) for e in prompt.split('\n')])
        try:
            if len(self.encoder.encode(prompt)) < 3800:
                sql = 'SELECT ' + self.llm.predict(prompt)
            else:
                sql = 'SELECT ' + self.llm_long.predict(prompt)
        except:
            if len(self.encoder.encode(prompt)) < 3800:
                self.llm.temperature = 0.2
                sql = 'SELECT ' + self.llm.predict(prompt)
                self.llm.temperature = 0
            else:
                self.llm_long.temperature = 0.2
                sql = 'SELECT ' + self.llm_long.predict(prompt)
                self.llm_long.temperature = 0
        sql = sql.replace("SELECT SELECT", "SELECT")
        sql = sql.replace('\n', ' ')
        sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
        return sql

def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents


def result_callback(result):
    exec_result.extend(result)


def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res, predicted_res[:10]


def execute_model(predicted_sql, ground_truth, schema, foreign_keys, question, knowledge, table_column_dict, db_place, idx, meta_time_out):
    try:
        res, result = func_timeout(meta_time_out, execute_sql,
                           args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(e,)]  # possibly len(query) > 512 or not executable
        res = 0
    # result = str(set([ret[0] for ret in result]))
    return_result = [{'sql_idx': idx, 'res': res, 'result': result, 'schema': schema, 'foreign_keys': foreign_keys,
                      'question': question, 'knowledge': knowledge, "table_column_dict": table_column_dict,
                      'pre': predicted_sql, 'gold': ground_truth}]
    # print(result)

    tmp_all_questions = [question]
    tmp_all_gold_sqls = [ground_truth]
    tmp_all_predict_sqls = [predicted_sql]
    tmp_all_knowledge = [knowledge]
    try:
        QAgenerator = QAGenerateTool()
        SQLgenerator = SQLGenerateTool()
        for j in range(0):
            new_question, new_gold_sql, new_knowledge = QAgenerator.run(tmp_all_questions[-1], schema, foreign_keys, tmp_all_gold_sqls[-1], tmp_all_knowledge[-1])
            tmp_all_questions.append(new_question)
            tmp_all_gold_sqls.append(new_gold_sql)
            tmp_all_knowledge.append(new_knowledge)

            new_sql = SQLgenerator.run(new_question, schema, foreign_keys, new_knowledge)
            tmp_all_predict_sqls.append(new_sql)

            # if g_flag is not False:
            #     print(set(p_result) == set(g_result))
            try:
                res, result = func_timeout(meta_time_out, execute_sql,
                                           args=(new_sql, new_gold_sql, db_place))
            except KeyboardInterrupt:
                sys.exit(0)
            except FunctionTimedOut:
                result = f'timeout'
                res = 0
            except Exception as e:
                result = e  # possibly len(query) > 512 or not executable
                res = 0

            return_result.append({'sql_idx': idx + 999999, 'res': res, 'result': result, 'schema': schema,
                                  'foreign_keys': foreign_keys, 'question': question, 'knowledge': knowledge,
                                  "table_column_dict": table_column_dict, 'pre': predicted_sql, 'gold': ground_truth})

    finally:
        return return_result


def  package_sqls(sql_path, db_root_path, mode='generate'):
    clean_sqls = []
    db_path_list = []
    if mode == 'generate':
        sqls = open(sql_path)
        sql_txt = sqls.readlines()
        for idx, sql_str in enumerate(sql_txt):
            sql = sql_str.strip()
            clean_sqls.append(sql)

    elif mode == 'gt':
        sqls = open(sql_path)
        sql_txt = sqls.readlines()
        print(sql_path)
        for idx, sql_str in enumerate(sql_txt):
            sql = '\t'.join(sql_str.strip().split('\t')[:-1])
            db_name = sql_str.strip().split('\t')[-1]
            clean_sqls.append(sql)
            db_path_list.append(db_root_path  + '/' + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list


def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth, schema, foreign_keys, question, knowledge, table_column_dict = sql_pair

        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, schema, foreign_keys, question, knowledge,
                                              table_column_dict, db_places[i], i, meta_time_out),
                         callback=result_callback)
    pool.close()
    pool.join()


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])


CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')

JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('-', '+', "*", '/')
AGG_OPS = ('max', 'min', 'count', 'sum', 'avg')
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


def eval_hard(sql):
    sql = sql.lower()
    last_sql = sql
    while sql != last_sql:
        sql = sql.replace('  ', ' ')
        last_sql = sql
    sql = sql.replace('(*)', '').replace('( * )', '')
    sql_split = sql.split()

    tables = []
    for i in range(len(sql_split)):
        if sql_split[i] == 'from' or sql_split[i] == 'join':
            tables.append(sql_split[i + 1])

    where_count = 0
    unit_count = 0
    agg_count = 0
    cond_count = 0
    sql_count = 0
    order_count = 0

    for where_ops in WHERE_OPS:
        if where_ops in sql:
            where_count += sql.count(where_ops)
    for unit_ops in UNIT_OPS:
        if unit_ops in sql:
            unit_count += sql.count(unit_ops)
    for agg_ops in AGG_OPS:
        if agg_ops in sql:
            agg_count += sql.count(agg_ops)
    for cond_ops in COND_OPS:
        if cond_ops in sql:
            cond_count += sql.count(cond_ops)
    for sql_ops in SQL_OPS:
        if sql_ops in sql:
            sql_count += sql.count(sql_ops)
    for order_ops in ORDER_OPS:
        if order_ops in sql:
            order_count += sql.count(order_ops)

    all_count = where_count + unit_count + agg_count + cond_count + sql_count + order_count

    tables = set(tables)

    if len(tables) == 1 and all_count <= 6:
        return 'simple'
    if len(tables) > 1 and all_count <= 6:
        return 'challenging'
    if len(tables) == 1 and all_count > 6:
        return 'moderate'
    return 'challenging'


def compute_acc_by_diff(exec_results, diff_json_path):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results, more_challenging_results = [], [], [], []

    for i, content in enumerate(contents):
        #########################
        content['difficulty'] = eval_hard(content['SQL'] if 'SQL' in content.keys() else content['query'])
        #########################

    simple_acc = sum([res['res'] for res in simple_results]) / len(simple_results) if len(simple_results) != 0 else 0
    moderate_acc = sum([res['res'] for res in moderate_results]) / len(moderate_results) if len(
        moderate_results) != 0 else 0
    challenging_acc = sum([res['res'] for res in challenging_results]) / len(challenging_results) if len(
        challenging_results) != 0 else 0
    # more_challenging_acc = sum([res['res'] for res in more_challenging_results]) / len(more_challenging_results)
    all_acc = sum(results) / num_queries
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    # count_lists = [len(simple_results), len(moderate_results), len(challenging_results), len(more_challenging_results) , num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists
    # return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, more_challenging_acc * 100, all_acc * 100, count_lists


def print_data(score_lists, count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))


def get_information(preprocessed_path, data_path):
    preprocessed_data = json.load(open(preprocessed_path))
    data = json.load(open(data_path))
    schemas = []
    all_foreign_keys = []
    questions = []
    all_knowledge = []
    all_table_column_list = []
    for i in range(len(preprocessed_data)):
        question = data[i]['question']
        knowledge = data[i].get('evidence')
        foreign_keys = generate_foreign_key(preprocessed_data[i])
        schema = generate_schema(preprocessed_data[i])
        schemas.append(schema)
        all_foreign_keys.append(foreign_keys)
        questions.append(question)
        all_knowledge.append(knowledge)
        all_table_column_list.append(generate_table_column_list(preprocessed_data[i]))
    return schemas, all_foreign_keys, questions, all_knowledge, all_table_column_list

def get_train_data(results, output_path):
    expect = []
    excepts = []
    for idx, res in enumerate(results):
        tmp_truth = {}
        for t in res["table_column_dict"].keys():
            for c in res["table_column_dict"][t]:
                if t in res['gold'].lower() and c in res['gold'].lower():
                    if t not in tmp_truth:
                        tmp_truth[t] = [c]
                    else:
                        tmp_truth[t].append(c)
            if t not in tmp_truth and t in res['gold'].lower() and '*' in res['gold'].lower():
                tmp_truth[t] = res["table_column_dict"][t][:3]

        if tmp_truth == {}:
            excepts.append(idx)
            continue

        if res['knowledge'] is not None:
            tmp = {
                'input': prompt_with_knowledge_result.format(question=res['question'],
                                                             schema=res['schema'],
                                                             foreign_keys=res['foreign_keys'],
                                                             knowledge=res['knowledge'],
                                                             result=res['result'],
                                                      SQL=res['pre']),
                'output': res['pre'] if res['res'] == 1 else res['gold']
            }
        else:
            tmp = {
                'input': prompt_without_knowledge_result.format(question=res['question'],
                                                             schema=res['schema'],
                                                             foreign_keys=res['foreign_keys'],
                                                             result=res['result'],
                                                             SQL=res['pre']),
                'output': res['pre'] if res['res'] == 1 else res['gold']
            }
        expect.append(tmp)
    print(excepts)
    with open(output_path, 'w') as f:
        json.dump(expect, f)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args_parser.add_argument('--diff_json_path', type=str, default='')
    args_parser.add_argument('--output_path', type=str, required=True, default='')
    args_parser.add_argument('--preprocessed_path', type=str, required=True, default='')
    args_parser.add_argument('--data_path', type=str, required=True, default='')

    args = args_parser.parse_args()

    if os.path.exists(args.output_path):
        sys.exit()

    exec_result = []

    pred_queries, _ = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict)
    # generate gt sqls:
    gt_queries, db_paths = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt')

    schemas, foreign_keys, questions, all_knowledge, all_table_column_list = get_information(args.preprocessed_path, args.data_path)

    query_pairs = list(zip(pred_queries, gt_queries, schemas, foreign_keys, questions, all_knowledge, all_table_column_list))
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
    get_train_data(exec_result, args.output_path)

    print('start calculate')
    # simple_acc, moderate_acc, challenging_acc, more_challenging_acc, acc, count_lists = \
    #     compute_acc_by_diff(exec_result,args.diff_json_path)
    # score_lists = [simple_acc, moderate_acc, challenging_acc, more_challenging_acc, acc]
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
        compute_acc_by_diff(exec_result, args.diff_json_path)
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("Finished evaluation")
