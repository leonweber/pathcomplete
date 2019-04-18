import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd


TYPE_MAPPING = {
    'Binding': 'in-complex-with',
    'Catalysis of acetylation': 'controls-state-change-of',
    'Catalysis of glycosylation': 'controls-state-change-of',
    'Catalysis of hydroxylation': 'controls-state-change-of',
    'Catalysis of methylation': 'controls-state-change-of',
    'Catalysis of phosphorylation': 'controls-phosphorylation-of',
    'Catalysis of ubiquitination': 'controls-state-change-of',
    'Regulation of expression': 'controls-expression-of',
    'Regulation of phosphorylation': 'controls-phosphorylation-of',
    'Regulation of localization': 'controls-transport-of',
    'Regulation of transcription': 'controls-expression-of'
}


def convert_to_sifnx(relations: pd.DataFrame, articles: List[str]) -> List[Dict]:
    for relation in relations.iterrows():
        if relation['refined_type'] not in TYPE_MAPPING:
            continue

        sifnx_type = TYPE_MAPPING[relation['refined_type']]






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--relations', required=True)
    parser.add_argument('--articles', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    relations = pd.read_csv(args.relations, sep='\t')
    with open(args.articles) as f:
        articles = f.read().splitlines(keepends=False)
    preds = convert_to_sifnx(relations, articles)
    print(preds)




