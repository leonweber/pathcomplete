import argparse

from pathlib import Path

# def filter_data(data, blacklisted_triples):


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--filter-data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)

    args = parser.parse_args()

