import argparse
import string
from collections import defaultdict
import sys
import itertools

import gensim
import numpy as np
import sent2vec

import pybel
from flair.tokenization import SegtokTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

from nltk.corpus import stopwords

MAX_EDIT_DIST = 10

STOPWORDS = set(i.lower() for i in stopwords.words("english") + list(string.punctuation) + ["also"])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bel", required=True)
    parser.add_argument("--dict", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bert", required=True)

    args = parser.parse_args()

    embeddings = sent2vec.Sent2vecModel()
    embeddings.load_model("/vol/fob-wbia-vol2/wbi/resources/embeddings/BioSentVec_PubMed_MIMICIII-bigram_d700.bin")
    graph = pybel.load(args.bel)

    cuid_to_syns = defaultdict(set)
    with open(args.dict) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cuids, syn = line.split("||")
            for cuid in cuids.split("|"):
                cuid_to_syns[cuid].add(syn)

    # aligner = SentenceAligner(model=args.bert)
    bert = BertModel.from_pretrained(args.bert)
    tokenizer = BertTokenizerFast.from_pretrained(args.bert)
    with open(args.out, "w") as f:
        for edge in tqdm(graph.edges):
            for e in edge[:2]:
                if not hasattr(e, "name") or e.name not in cuid_to_syns:
                    continue
                evidence = graph.get_edge_evidence(*edge).lower()

                best_match = find_best_match(cuid_to_syns[e.name], evidence)
                if best_match:
                    start, end = best_match
                    entity_type = type(e).__name__
                    result_str = f"1||{start}|{end}||{entity_type}||{evidence[start:end]}||{e.name}\n"
                    f.write(result_str)
                else:
                    print()
                    print(evidence)
                    print(cuid_to_syns[e.name])
