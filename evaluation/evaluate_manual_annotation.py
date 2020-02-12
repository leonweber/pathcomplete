import argparse
import math
from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
from sklearn.metrics import average_precision_score

def radar_chart(df, models, ax):

    categories = list(df)
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]


    # If you want the first axis to be on top:
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)

    values = df.loc[0].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=models[0])
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values = df.loc[1].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=models[1])
    ax.fill(angles, values, 'r', alpha=0.1)




def get_scores(path):
    try:
        data = np.loadtxt(path).reshape(5, 10)
    except ValueError:
        data = np.loadtxt(path).reshape(10, 10)


    aps = []

    for row in data:
        row = row[~np.isnan(row)]
        scores = 1/np.arange(1, len(row)+1)
        ap = average_precision_score(row, scores)
        if np.isnan(ap):
            ap = 0
        aps.append(ap)

    return {
        'mAP': np.mean(aps),
        'p@1': np.nanmean(data[:, :1]),
        'p@5': np.nanmean(data[:, :5]),
        'p@10': np.nanmean(data[:, :10]),
            }

def get_score_by_file(path):
    files = list(path.glob('*.tsv'))


    scores_by_file = {}

    for file in files:
        scores_by_file[file.name] = get_scores(file)

    return scores_by_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    args = parser.parse_args()

    sns.set_style('whitegrid')
    sns.set_palette("Set1")

    scores_by_file = get_score_by_file(args.input)
    relations = sorted(set(f.split('_')[0] for f in scores_by_file))
    models = sorted(set(f.split('_')[1].split('.')[0] for f in scores_by_file))


    for metric in ['mAP', 'p@1', 'p@5', 'p@10']:
        df = defaultdict(list)
        for model in models:
            for rel in relations:
                score = scores_by_file[f"{rel}_{model}.tsv"][metric]
                df[rel].append(score)

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        df = pd.DataFrame(df)
        radar_chart(df, models, ax)
        fig.savefig(args.input/f'{metric}.png', dpi=300)












