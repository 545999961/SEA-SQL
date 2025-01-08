import argparse
import json


def parse_option():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--bias_path1', type=str, default="")
    parser.add_argument('--bias_path2', type=str, default="")
    parser.add_argument('--schema_path1', type=str, default="")
    parser.add_argument('--schema_path2', type=str, default="")
    parser.add_argument('--output_bias_path', type=str, default="")
    parser.add_argument('--output_schema_path', type=str, default="")


    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_option()
    bias1 = json.load(open(opt.bias_path1))
    bias2 = json.load(open(opt.bias_path2))
    bias = bias1
    bias.extend(bias2)
    with open(opt.output_bias_path, 'w') as f:
        json.dump(bias, f)

    schema1 = json.load(open(opt.schema_path1))
    schema2 = json.load(open(opt.schema_path2))
    schema = schema1
    schema.extend(schema2)
    with open(opt.output_schema_path, 'w') as f:
        for s in schema:
            f.write(json.dumps(s) + '\n')