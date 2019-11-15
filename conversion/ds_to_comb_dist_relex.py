import argparse
import json
from pathlib import Path

def transform(data, direct_data):
    transformed_data = {}
    for k, v in data.items():
        v['supervision_type'] = 'distant'
        transformed_data[k] = v
    for k, v in direct_data.items():
        if k in transformed_data:
            k = k + '_direct'

        v['mentions'] = [m for m in v['mentions'] if m[1] == 'direct']
        if v['mentions']:
            v['supervision_type'] = 'direct'
            transformed_data[k] = v

    return transformed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--direct_data', type=Path)

    args = parser.parse_args()

    with args.input.open() as f:
        data = json.load(f)

    if args.direct_data:
        with args.direct_data.open() as f:
            direct_data = json.load(f)
    else:
        direct_data = {}

    with args.output.open('w') as f:
        transformed_data = transform(data, direct_data=direct_data)
        json.dump(transformed_data, f)
