import sqlite3
import json
import os
import argparse

def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--database_path', type=str, default="dev.json")
    parser.add_argument('--table_path', type=str, default="dev.json")

    opt = parser.parse_args()

    return opt

def traverse_directory_using_os(root_folder):
    file_list = []
    if not os.path.isdir(root_folder):
        file_list.append(root_folder)
    else:
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                file_list.append(full_path)
    new_file_list = []
    for f in file_list:
        if '.sqlite' in f:
            new_file_list.append(f)
    return new_file_list

def extract_schema(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # 结果字典
    # 结果字典
    result = {
        "column_names": [],
        "column_names_original": [],
        "column_types": [],
        "db_id": db_file.split('/')[-1].split('.')[0],  # 取文件名作为 db_id
        "foreign_keys": [],
        "primary_keys": [],
        "table_names": [],
        "table_names_original": []
    }

    # 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    result["column_names"].append([-1, '*'])
    result["column_names_original"].append([-1, '*'])
    result["column_types"].append("text")
    for idx, table in enumerate(tables):
        table_name = table[0]
        result["table_names"].append(table_name)
        result["table_names_original"].append(table_name)

        # 获取列信息
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        # 暂存列信息
        table_column_names = []
        table_column_names_original = []
        table_column_types = []

        for col in columns:
            cid, name, type_, notnull, dflt_value, pk = col
            table_column_names.append(name)
            table_column_names_original.append(name)  # 假设原名与列名相同
            table_column_types.append(type_)

            # 更新列名和索引
            result["column_names"].append([idx, name])
            result["column_names_original"].append([idx, name])
            if pk:
                result["primary_keys"].append(len(result["column_names_original"]) - 1)

        result["column_types"].extend(table_column_types)

        # 获取外键
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        foreign_keys = cursor.fetchall()
        for fk in foreign_keys:
            if len(fk) >= 4:  # 确保有足够的元素
                table_from = fk[0]
                from_col = fk[3]
                table_to = fk[2]
                to_col = fk[4]
                result["foreign_keys"].append([table_name, from_col, table_to, to_col])
    
    for i in range(len(result["foreign_keys"])):
        from_table_idx = result["table_names"].index(result["foreign_keys"][i][0])
        to_table_idx = result["table_names"].index(result["foreign_keys"][i][2])

        from_column_idx, to_column_idx = 0, 0
        for j in range(len(result["column_names_original"])):
            if result["column_names_original"][j][0] == from_table_idx and result["column_names_original"][j][1] == result["foreign_keys"][i][1]:
                from_column_idx = j
            if result["column_names_original"][j][0] == to_table_idx and result["column_names_original"][j][1] == result["foreign_keys"][i][3]:
                to_table_idx = j
        result["foreign_keys"][i] = [from_column_idx, to_table_idx]

    conn.close()

    return result

    
def main(opt):
    path = opt.database_path
    all_paths = traverse_directory_using_os(path)
    results = []
    for p in all_paths:
        schema_info = extract_schema(p)
        results.append(schema_info)
        
    with open(opt.table_path, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    opt = parse_option()
    main(opt)
