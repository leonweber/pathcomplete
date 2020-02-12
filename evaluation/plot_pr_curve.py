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
    parser.add_argument('--relations', required=False, nargs='+')

    args = parser.parse_args()

    if args.model_names:
        assert len(args.model_names) == len(args.preds)
    else:
        args.model_names = [str(p) for p in args.preds]



    with args.anns.open() as f:
        anns = {k: v for k, v in json.load(f).items() if v['mentions']}

    df = None
    sent_df = {"ap": [], "Model": []}
    for preds, model in zip(args.preds, args.model_names):
        with preds.open() as f:
            preds = f.readlines()
            _, sent_aps, result_df = evaluate(anns, preds)

            sent_df["ap"] += sent_aps
            sent_df["Model"] += [model] * len(sent_aps)

            result_df['Model'] = model
            if df is None:
                df = result_df
            else:
                df = pd.concat([df, result_df])

    sent_df = pd.DataFrame(sent_df)

    if args.relations:
        df = df[df['Relation'].isin(args.relations)]


    sns.set()
    sns.set_style('whitegrid')

    try:
        ax = sns.barplot(x="Model", y="ap", data=sent_df, ci=None)
        ax.set(ylabel="sentential mAP")
        ax.get_figure().savefig(str(args.preds[0]) + '_sentential_ap.jpg', dpi=300)
    except ValueError:
        pass

    n_models = len(df['Model'].unique())
    g = sns.FacetGrid(df, col='Relation', legend_out=False, col_wrap=3, hue="Model",
                      # hue_kws={
                      #    'color': sns.color_palette()[:n_models],
                      #    # 'linestyle': ['-', "--", "-.", ":"][:n_models]
                      #    #  'marker': ['o', '^', 's', 'P'][:n_models],
                      # },
                      )
    g.map(plt.plot, 'Recall', 'Precision', markevery=0.3)
    if not args.relations or len(args.relations) > 1:
        g.set_titles(col_template="{col_name}")
    else:
        g.set_titles(" ")
    # g.add_legend()
    if not args.relations or len(args.relations) > 1:
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper right')

    g.savefig(str(args.preds[0]) + '_pr.jpg', dpi=300, bbox_inches='tight')







