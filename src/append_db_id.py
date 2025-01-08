import argparse
import json

def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--dev_path', type=str, default="dev.json")
    parser.add_argument('--input_path', type=str, default="../intermediate_datasets_bird/third_round.sql")
    parser.add_argument('--output_path', type=str, default="../intermediate_datasets_bird/predict_dev.json")

    opt = parser.parse_args()

    return opt

def main(opt):
    dev_path = opt.dev_path
    output_path = opt.output_path
    sql_path = opt.input_path

    dev = json.load(open(dev_path))
    sqls = []
    with open(sql_path) as f:
        lines = f.readlines()
        for line in lines:
            sqls.append(line.strip())

    for i in range(len(sqls)):
        # sqls[i] = sqls[i] + '\t----- bird -----\t' + dev[i]['db_id']
        sqls[i] = sqls[i].replace("|| ', ' ||", ", ").replace("|| ' ' ||", ", ") + '\t----- bird -----\t' + dev[i]['db_id']


    result = {}
    for i, sql in enumerate(sqls):
        result[i] = sql

    if output_path:
        json.dump(result, open(output_path, 'w'), indent=4)


if __name__ == "__main__":
    opt = parse_option()
    main(opt)