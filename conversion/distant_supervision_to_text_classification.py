import argparse
import json
from pathlib import Path
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    df = {'text': [], 'label': [], 'pair': [], 'pmid': []}
    with args.input.open() as f:
        data = json.load(f)
    for k, v in data.items():
        for text, supervision_type, pmid in v['mentions']:
            df['text'].append(text)
            df['label'].append(supervision_type == 'direct')
            df['pair'].append(k)
            df['pmid'].append(pmid)

    df = pd.DataFrame(df)
    df.to_csv(args.output)
    with (args.output.parent/'classes.txt').open('w') as f:
        f.write('\n'.join(df['label'].unique()))


