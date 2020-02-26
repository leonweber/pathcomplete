import argparse
import json
from pathlib import Path
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    df = {'text': [], 'labels': [], 'pair': [], 'pmid': []}
    with args.input.open() as f:
        data = json.load(f)
    for k, v in data.items():
        for text, supervision_type, pmid in v['mentions']:
            df['text'].append(text)
            df['labels'].append(','.join(v['relations']))
            df['pair'].append(k)
            df['pmid'].append(pmid)

    df = pd.DataFrame(df)
    df.to_csv(args.output)


