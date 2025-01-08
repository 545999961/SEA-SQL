import copy
import os
import sqlite3
import sys
from itertools import combinations
import re
from sql_metadata import Parser


def generate_foreign_key(data):
    fk_str = ''
    for fk in data['fk']:
        fk_str += ('# ' + fk['source_table_name_original'] + '.' + fk['source_column_name_original'] + ' = '
                   + fk['target_table_name_original'] + '.' + fk['target_column_name_original'] + '\n')
    return fk_str[:-1]


def generate_schema(data):
    schema = ""
    for table in data['db_schema']:
        schema += '# ' + table['table_name_original'] + ' ( '
        for i, column in enumerate(table['column_names_original']):
            schema += f'{column} ('
            left_parenthesis = False
            # if table['column_description'][i] != '':
            #     schema += f"column description: {table['column_description'][i]}, "
            #     left_parenthesis = True
            # if table['value_description'][i] != '':
            #     schema += f"value description: {table['value_description'][i]}, "
            #     left_parenthesis = True
            if table['db_contents'][i]:
                value_flag = False
                schema += ' e.g., `'
                for value in table['db_contents'][i]:
                    if len(str(value)) < 100:
                        schema += str(value) + '`, `'
                        value_flag = True
                if value_flag:
                    schema = schema[: -3] + ', etc. )'
                elif left_parenthesis:
                    schema = schema[: -8] + ' )'
                else:
                    schema = schema[: -10]
            schema += ', '
        schema = schema[:-2] + ' )\n'
    return schema[:-1]

def generate_schema_simple(data):
    schema = ""
    for table in data['db_schema']:
        schema += '# ' + table['table_name_original'] + ' ( '
        for i, column in enumerate(table['column_names_original']):
            schema += f'{column} ('
            # left_parenthesis = False
            # if table['column_description'][i] != '':
            #     schema += f"column description: {table['column_description'][i]}, "
            #     left_parenthesis = True
            # if table['value_description'][i] != '':
            #     schema += f"value description: {table['value_description'][i]}, "
            #     left_parenthesis = True
            # if table['db_contents'][i]:
            #     value_flag = False
            #     schema += ' e.g., `'
            #     for value in table['db_contents'][i]:
            #         if len(str(value)) < 100:
            #             schema += str(value) + '`, `'
            #             value_flag = True
            #     if value_flag:
            #         schema = schema[: -3] + ', etc. )'
            #     elif left_parenthesis:
            #         schema = schema[: -8] + ' )'
            #     else:
            #         schema = schema[: -10]
            schema += ', '
        schema = schema[:-2] + ' )\n'
    return schema[:-1]


def generate_foreign_key_by_tables(data, uses_tables):
    fk_str = ''
    for fk in data['fk']:
        if fk['source_table_name_original'] in uses_tables and fk['target_table_name_original'] in uses_tables:
            fk_str += ('# ' + fk['source_table_name_original'] + '.' + fk['source_column_name_original'] + ' = '
                       + fk['target_table_name_original'] + '.' + fk['target_column_name_original'] + '\n')
    return fk_str[:-1]


def generate_schema_by_tables(data, uses_tables):
    schema = ""
    for table in data['db_schema']:
        if table['table_name_original'] in uses_tables:
            schema += '# ' + table['table_name_original'] + ' ( '
            for i, column in enumerate(table['column_names_original']):
                schema += column
                if table['db_contents'][i]:
                    value_flag = False
                    schema += ' ( e.g., `'
                    for value in table['db_contents'][i]:
                        if len(str(value)) < 100:
                            schema += str(value) + '`, `'
                            value_flag = True
                    if value_flag:
                        schema = schema[:-3] + ', etc. )'
                    else:
                        schema = schema[:-10]
                schema += ', '
            schema = schema[:-2] + ' )\n'
    return schema[:-1]


def get_subsets(lst):
    subsets = []
    for r in range(2, len(lst) + 1):
        subsets.extend(combinations(lst, r))
    return subsets


def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            new_path = find_path(graph, node, end, path)
            if new_path:
                return new_path
    return None


def get_tables(sql, data):
    sql = sql.lower()
    all_tables = [table['table_name_original'] for table in data['db_schema']]
    sql_tokens = [s for s in sql.replace("  ", " ").split(" ") if s.strip() != ""]
    uses_tables = []
    end_punctuations = [s[-1] for s in all_tables if s and not s[-1].isalnum()]
    sql_tokens = [s if (s[-1].isalnum() or s[-1] in end_punctuations) else s[:-1] for s in sql_tokens]

    for table in all_tables:
        if table in sql_tokens:
            uses_tables.append(table)
    if len(uses_tables) == 0:
        uses_tables = all_tables

    new_tables = copy.deepcopy(uses_tables)
    for fk in data['fk']:
        if fk['source_table_name_original'] in uses_tables:
            new_tables.append(fk['target_table_name_original'])
        elif fk['target_table_name_original'] in uses_tables:
            new_tables.append(fk['source_table_name_original'])
    uses_tables = list(set(new_tables))

    table_fk = list()
    for fk in data['fk']:
        source = fk['source_table_name_original']
        target = fk['target_table_name_original']
        table_fk.append([source, target])

    graph = dict()
    for edge in table_fk:
        node1, node2 = edge
        if node1 in graph:
            graph[node1].append(node2)
        else:
            graph[node1] = [node2]
        if node2 in graph:
            graph[node2].append(node1)
        else:
            graph[node2] = [node1]

    if len(uses_tables) < 10:
        table_subsets = get_subsets(uses_tables)
        for pair in table_subsets:
            first = pair[0]
            second = pair[1]
            pair_flag = True
            for fk in table_fk:
                if first in fk and second in fk:
                    pair_flag = False
            if pair_flag:
                path = find_path(graph, first, second)
                if path:
                    for p in path:
                        uses_tables.append(p)
                else:
                    for fk in table_fk:
                        if first in fk and second not in fk:
                            uses_tables.append(second)
                        elif first not in fk and second in fk:
                            uses_tables.append(first)

    return list(set(uses_tables))


def get_tables_from_dict(uses_tables, data):
    new_tables = copy.deepcopy(uses_tables)
    for fk in data['fk']:
        if fk['source_table_name_original'] in uses_tables:
            new_tables.append(fk['target_table_name_original'])
        elif fk['target_table_name_original'] in uses_tables:
            new_tables.append(fk['source_table_name_original'])
    uses_tables = list(set(new_tables))

    table_fk = list()
    for fk in data['fk']:
        source = fk['source_table_name_original']
        target = fk['target_table_name_original']
        table_fk.append([source, target])

    graph = dict()
    for edge in table_fk:
        node1, node2 = edge
        if node1 in graph:
            graph[node1].append(node2)
        else:
            graph[node1] = [node2]
        if node2 in graph:
            graph[node2].append(node1)
        else:
            graph[node2] = [node1]

    if len(uses_tables) < 10:
        table_subsets = get_subsets(uses_tables)
        for pair in table_subsets:
            first = pair[0]
            second = pair[1]
            pair_flag = True
            for fk in table_fk:
                if first in fk and second in fk:
                    pair_flag = False
            if pair_flag:
                path = find_path(graph, first, second)
                if path:
                    for p in path:
                        uses_tables.append(p)
                else:
                    for fk in table_fk:
                        if first in fk and second not in fk:
                            uses_tables.append(second)
                        elif first not in fk and second in fk:
                            uses_tables.append(first)

    return list(set(uses_tables))

def normalization(sql):
    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])

        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                if in_quotation:
                    in_quotation = False
                else:
                    in_quotation = True

        return out_s

    # remove ";"
    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation
    def double2single(s):
        return s.replace("\"", "'")

    def add_asc(s):
        pattern = re.compile(
            r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")

        return s

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]

        tables_aliases = new_tables_aliases
        for k, v in tables_aliases.items():
            s = s.replace("as " + k + " ", "")
            s = s.replace(k, v)

        return s

    processing_func = lambda x: remove_table_alias(add_asc(lower(white_space_fix(double2single(remove_semicolon(x))))))

    return processing_func(sql)


def get_table_columns(sql, data):
    norm_sql = normalization(sql).strip()
    sql_tokens = norm_sql.split()
    uses_dict = {}
    for d in data['db_schema']:
        if d['table_name_original'] in sql_tokens:
            uses_dict[d['table_name_original']] = []
            for column_name_original in d["column_names_original"]:
                if column_name_original in sql_tokens or \
                        d["table_name_original"] + "." + column_name_original in sql_tokens:
                    uses_dict[d['table_name_original']].append(column_name_original)
    return uses_dict


def get_tables_columns_dict(sql, data):
    sql = sql.lower()
    all_tables = [table['table_name_original'] for table in data['db_schema']]
    uses_tables = []
    return_dict = {}

    for table in all_tables:
        if table in sql:
            uses_tables.append(table)
    if len(uses_tables) == 0:
        uses_tables = all_tables

    for d in data['db_schema']:
        table = d['table_name_original']
        if table in uses_tables:
            return_dict[table] = []

    for fk in data['fk']:
        if fk['source_table_name_original'] in uses_tables:
            if fk['source_column_name_original'] not in return_dict[fk['source_table_name_original']]:
                return_dict[fk['source_table_name_original']].append(fk['source_column_name_original'])
            if fk['target_table_name_original'] not in uses_tables:
                return_dict[fk['target_table_name_original']] = []
            if fk['target_column_name_original'] not in return_dict[fk['target_table_name_original']]:
                return_dict[fk['target_table_name_original']].append(fk['target_column_name_original'])
        elif fk['target_table_name_original'] in uses_tables:
            if fk['target_column_name_original'] not in return_dict[fk['target_table_name_original']]:
                return_dict[fk['target_table_name_original']].append(fk['target_column_name_original'])
            if fk['source_table_name_original'] not in uses_tables:
                return_dict[fk['source_table_name_original']] = []
            if fk['source_column_name_original'] not in return_dict[fk['source_table_name_original']]:
                return_dict[fk['source_table_name_original']].append(fk['source_column_name_original'])

    for d in data['db_schema']:
        table = d['table_name_original']
        columns = d["column_names_original"]
        if table in return_dict.keys():
            for column in columns:
                if column in sql:
                    return_dict[table].append(column)
            if len(return_dict[table]) == 0:
                return_dict[table] = columns
            return_dict[table] = list(set(return_dict[table]))

    return return_dict

def get_tables_columns_dict_only(sql, data):
    sql = sql.lower()
    all_tables = [table['table_name_original'] for table in data['db_schema']]
    uses_tables = []
    return_dict = {}

    for table in all_tables:
        if table in sql:
            uses_tables.append(table)
    if len(uses_tables) == 0:
        uses_tables = all_tables

    for d in data['db_schema']:
        table = d['table_name_original']
        if table in uses_tables:
            return_dict[table] = []

    for d in data['db_schema']:
        table = d['table_name_original']
        columns = d["column_names_original"]
        if table in return_dict.keys():
            for column in columns:
                if column in sql:
                    return_dict[table].append(column)
            if len(return_dict[table]) == 0:
                return_dict[table] = columns
            return_dict[table] = list(set(return_dict[table]))

    return return_dict

def generate_schema_by_dict_only(data, use_dict):
    schema = ""
    for table in data['db_schema']:
        if table['table_name_original'] in use_dict.keys():
            flag = False
            schema += '# ' + table['table_name_original'] + ' ( '
            for i, column in enumerate(table['column_names_original']):
                if column not in use_dict[table['table_name_original']]:
                    continue
                schema += f'{column} ('
                left_parenthesis = False
                # if table['column_description'][i] != '':
                #     schema += f"column description: {table['column_description'][i]}, "
                #     left_parenthesis = True
                # if table['value_description'][i] != '':
                #     schema += f"value description: {table['value_description'][i]}, "
                #     left_parenthesis = True
                if table['db_contents'][i]:
                    value_flag = False
                    schema += ' e.g., `'
                    for value in table['db_contents'][i]:
                        if len(str(value)) < 100:
                            schema += str(value) + '`, `'
                            value_flag = True
                    if value_flag:
                        schema = schema[: -3] + ', etc. )'
                    elif left_parenthesis:
                        schema = schema[: -8] + ' )'
                    else:
                        schema = schema[: -10]
            if flag:
                schema = schema[:-2] + ' )\n'
            else:
                schema = schema[:-3] + '\n'
        # else: # 没用到的表
        #     flag = False
        #     schema += '# ' + table['table_name_original'] + ' ( '
        #     for i, column in enumerate(table['column_names_original']):
        #         schema += column
        #         schema += ', '
        #         flag = True
        #     if flag:
        #         schema = schema[:-2] + ' )\n'
        #     else:
        #         schema = schema[:-3] + '\n'
    return schema[:-1]


def run_sql(db, sql):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        p_res = cursor.fetchall()
        return None
    except Exception as e:
        return e

import queue
import threading
import sqlite3

class QueryThread(threading.Thread):
    def stop(self):
        lock = self._tstate_lock
        if lock is not None:
            assert not lock.locked()
        self._is_stopped = True
        self._tstate_lock = None
        if not self.daemon:
            import _thread
            _allocate_lock = _thread.allocate_lock
            _shutdown_locks_lock = _allocate_lock()
            _shutdown_locks = set()
            with _shutdown_locks_lock:
                _shutdown_locks.discard(lock)

def new_run_sql(db, sql):
    # 定义一个队列用于存储查询结果或异常信息
    result_queue = queue.Queue()

    # 定义一个函数用于执行查询语句，并将结果或异常信息放入队列中
    def execute_query():
        try:
            # tmp_sql = sql
            # if 'limit' not in tmp_sql.lower():
            #     tmp_sql = tmp_sql.replace('')
            conn = sqlite3.connect(db)
            cursor = conn.cursor()
            cursor.execute(sql)
            p_res = cursor.fetchall()
            result_queue.put((p_res[:10], True))
        except Exception as e:
            result_queue.put((e, False))

    # 创建一个线程用于执行查询语句
    query_thread = QueryThread(target=execute_query)

    # 启动线程
    query_thread.start()

    # 等待线程执行完毕或超时
    query_thread.join(timeout=30)

    # 如果线程仍在执行，则终止线程并返回超时信息
    if query_thread.is_alive():
        return 'Execution timeout', False

    # 否则从队列中获取查询结果或异常信息，并返回
    return result_queue.get()

# print(new_run_sql('/home/baaiks/cf/pycharm/DAMO-ConvAI/bird/llm/data/dev/dev_databases/card_games/card_games.sqlite',
#               "SELECT COUNT(*)  FROM cards  WHERE rarity = 'rare'  AND types = 'Enchantment'  AND name = 'Abundance'  AND id IN (     SELECT uuid      FROM legalities      WHERE status = 'Legal'      GROUP BY uuid      HAVING COUNT(*) = (         SELECT COUNT(*)          FROM legalities          WHERE uuid = cards.id     ) )"))
# sys.exit(1)
# print(1)
def generate_schema_list(data):
    schema = ""
    num = 1
    for table in data['db_schema']:
        for i, column in enumerate(table['column_names_original']):
            schema += f"# [{num}]. {table['table_name_original']}.{column}\n"
            num += 1
    return schema[:-1]


def generate_schema_list_all(data, knowledge: None):
    schema = []
    for table in data['db_schema']:
        for i, column in enumerate(table['column_names_original']):
            temp_schema = ''
            temp_schema += f"table: {table['table_name_original']}, column: {column}"
            if table['column_description'][i] != '':
                temp_schema += f", column description: {table['column_description'][i]}"
            if table['value_description'][i] != '':
                temp_schema += f", value description: {table['value_description'][i]}"
            if table['db_contents'][i]:
                use_flag = False
                temp_schema += ", sample value: "
                for value in table['db_contents'][i]:
                    if len(str(value)) < 100:
                        temp_schema += f'{str(value)}, '
                        use_flag = True
                if use_flag:
                    temp_schema = temp_schema[:-2]
                else:
                    temp_schema = temp_schema[:-16]
            if knowledge is not None and column in knowledge:
                temp_schema += f', external knowledge: {knowledge}'

            schema.append(temp_schema)
    return schema

def generate_schema_by_dict_sort(data, use_dict):
    schema = ""
    for table in use_dict.keys():
        schema += '# ' + table + ' ( '
        for column in use_dict[table]:
            schema += column
            for table_source in data['db_schema']:
                if table_source['table_name_original'] == table:
                    for i, column_source in enumerate(table_source['column_names_original']):
                        if column_source == column and table_source['db_contents'][i]:
                            schema += ' ( e.g., `'
                            for value in table_source['db_contents'][i]:
                                if len(str(value)) < 100:
                                    schema += str(value) + '`, `'
                            schema = schema[:-3] + ', etc. )'
                            # break
                # break
            schema += ', '
        schema = schema[:-2] + ' )\n'

    return schema[:-1]

def get_foreign_keys_list(data):
    foreign_dict = {}
    for d in data['fk']:
        if d['source_table_name_original'] not in foreign_dict:
            foreign_dict[d['source_table_name_original']] = [d['source_column_name_original']]
        else:
            foreign_dict[d['source_table_name_original']].append(d['source_column_name_original'])
        if d['target_table_name_original'] not in foreign_dict:
            foreign_dict[d['target_table_name_original']] = [d['target_column_name_original']]
        else:
            foreign_dict[d['target_table_name_original']].append(d['target_column_name_original'])

    for table in data['db_schema']:
        s_t = table['table_name_original']
        for column in table['column_names_original']:
            for t_table in data['db_schema']:
                t_t = t_table['table_name_original']
                if s_t != t_t:
                    for t_column in t_table['column_names_original']:
                        if column == t_column:
                            if s_t not in foreign_dict:
                                foreign_dict[s_t] = [column]
                            else:
                                foreign_dict[s_t].append(column)
                            if t_t not in foreign_dict:
                                foreign_dict[t_t] = [t_column]
                            else:
                                foreign_dict[t_t].append(t_column)

    return foreign_dict

def get_primary_keys_list(data):
    primary_keys_dict = {}
    for d in data['pk']:
        primary_keys_dict[d['table_name_original']] = [d['column_name_original']]
    return primary_keys_dict