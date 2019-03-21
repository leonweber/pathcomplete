import argparse
from pathlib import Path

from sklearn import model_selection

IGNORE = {'has_subreaction'}
DONT_SPLIT = {'has_id'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()


    with open(args.input) as f:
        common_lines = []
        split_lines = []

        for line in f:
            triple = line.strip().split()
            if triple[2] in IGNORE:
                continue

            if triple[2] in DONT_SPLIT:
                common_lines.append(line)
            else:
                split_lines.append(line)

    train_lines, split_lines = model_selection.train_test_split(split_lines, train_size=0.5)
    dev_lines, test_lines = model_selection.train_test_split(split_lines, test_size=0.5)

    with open(args.input + '.train', 'w')  as f:
        f.writelines(train_lines + common_lines)
    with open(args.input + '.dev', 'w')  as f:
        f.writelines(dev_lines)
    with open(args.input + '.test', 'w')  as f:
        f.writelines(test_lines)










