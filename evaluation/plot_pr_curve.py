import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

from evaluate_distant_supervision import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns', required=True, type=Path)
    parser.add_argument('--preds', required=True, type=Path, nargs='+')
    parser.add_argument('--model_names', required=False, nargs='+')

    args = parser.parse_args()

    if args.model_names:
        assert len(args.model_names) == len(args.preds)
    else:
        args.model_names = [str(p) for p in args.preds]


    with args.anns.open() as f:
        anns = {k: v for k, v in json.load(f).items() if v['mentions']}

    df = None
    sent_df = {"ap": [], "model": []}
    for preds, model in zip(args.preds, args.model_names):
        with preds.open() as f:
            preds = f.readlines()
            _, sent_aps, result_df = evaluate(anns, preds)

            sent_df["ap"] += sent_aps
            sent_df["model"] += [model] * len(sent_aps)

            result_df['model'] = model
            if df is None:
                df = result_df
            else:
                df = pd.concat([df, result_df])

    sent_df = pd.DataFrame(sent_df)


    sns.set_style('whitegrid')
    sns.set_palette("deep")

    ax = sns.barplot(x="model", y="ap", data=sent_df, ci=None)
    ax.set(ylabel="sentential mAP")
    ax.get_figure().savefig(str(args.preds[0]) + '_sentential_ap.jpg', dpi=300)

    g = sns.FacetGrid(df, col='relation', hue='model', legend_out=True, col_wrap=3)
    g.map(sns.lineplot, 'recall', 'precision', ci=None)
    g.set_titles(col_template="{col_name}")
    plt.legend()
    g.savefig(str(args.preds[0]) + '_pr.jpg', dpi=300)






