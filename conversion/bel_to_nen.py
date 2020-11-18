import argparse
import csv
import re
import string
from collections import defaultdict
import sys
import itertools
import io

import gensim
import numpy as np
import sent2vec

import pybel
from flair.models import MultiTagger
from flair.tokenization import SegtokTokenizer, SciSpacySentenceSplitter, SegtokSentenceSplitter
from fuzzywuzzy import process
from fuzzywuzzy.fuzz import QRatio
from lxml import etree
from pybel import BELGraph
from pybel.io.line_utils import parse_lines, parse_statements
from pybel.parser.parse_bel import BELParser
from sklearn.metrics.pairwise import cosine_similarity
from conversion.utils import find_matches_sent2vec

from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

from nltk.corpus import stopwords

MAX_EDIT_DIST = 10

STOPWORDS = set(i.lower() for i in stopwords.words("english") + list(string.punctuation) + ["also"])

METADATA = """
DEFINE NAMESPACE CHEBI          AS URL "https://raw.githubusercontent.com/OpenBEL/openbel-framework-resources/20150611/namespace/chebi.belns"
DEFINE NAMESPACE MESHD          AS URL "https://raw.githubusercontent.com/OpenBEL/openbel-framework-resources/20150611/namespace/mesh-diseases.belns"
DEFINE NAMESPACE GOBP           AS URL "https://raw.githubusercontent.com/OpenBEL/openbel-framework-resources/20150611/namespace/go-biological-process.belns"
DEFINE NAMESPACE MGI            AS URL "https://raw.githubusercontent.com/OpenBEL/openbel-framework-resources/20150611/namespace/mgi-mouse-genes.belns"
DEFINE NAMESPACE HGNC           AS URL "https://raw.githubusercontent.com/OpenBEL/openbel-framework-resources/20150611/namespace/hgnc-human-genes.belns"
"""




def get_mentions(agents, text, synonyms_by_db, tagger):
    id_to_mentions = defaultdict(list)
    sents = sentence_splitter.split(text)
    tagger.predict(sents)
    genes = []
    chemicals = []
    for sent in sents:
        sent_genes = sent.get_spans("hunflair-gene")
        for i in sent_genes:
            i.start_pos += sent.start_pos
            i.end_pos += sent.start_pos
        genes += sent_genes

        sent_chemicals = []
        for i in sent_chemicals:
            i.start_pos += sent.start_pos
            i.end_pos += sent.end_pos
        chemicals += sent_chemicals

    mentions = genes + chemicals
    id_to_mention_scores = []
    agent_to_id = {}
    for agent in agents:
        for db, id_ in agent.db_refs.items():
            if db in synonyms_by_db:
                # syns = [Sentence(i) for i in synonyms_by_db[db][id_]]
                agent_to_id[agent] = (db, id_)
                break
    ids = [(db, id) for db, id in agent_to_id.values()]
    for db, id in ids:
        syns = synonyms_by_db[db][id]
        if len(syns) == 0:
            mention_scores = [-float('inf')] * len(mentions)
        else:
            mention_scores = [process.extractOne(i.text, syns)[1] for i in mentions]
        id_to_mention_scores.append(mention_scores)
    id_to_mention_scores = np.array(id_to_mention_scores)
    if id_to_mention_scores.shape[1] == 0:
        return {}
    for i_id, id in enumerate(ids):
        max_mention_idx = id_to_mention_scores[i_id].argmax()
        if id_to_mention_scores[:, max_mention_idx].argmax() == i_id and not np.isinf(id_to_mention_scores[i_id, max_mention_idx]):
            max_mention = mentions[max_mention_idx]
            id_to_mentions[id].append((max_mention.start_pos, max_mention.end_pos))

    return id_to_mentions


def get_mentions_hunflair(id_, text, cuid_to_synonyms, tagger, sentence_splitter):
    sents = sentence_splitter.split(text)
    tagger.predict(sents)
    if id_.startswith("CHEBI:"):
        span_key = "hunflair-chemical"
    elif id_.startswith("HGNC:") or id_.startswith("MGI:") or id_.startswith("EGID:"):
        span_key = "hunflair-gene"
    elif id_.startswith("MESHD:"):
        span_key = "hunflair-disease"
    else:
        raise ValueError(id_)
    mentions = []
    for sent in sents:
        sent_genes = sent.get_spans(span_key)
        for i in sent_genes:
            i.start_pos += sent.start_pos
            i.end_pos += sent.start_pos
        mentions += sent_genes
    syns = cuid_to_synonyms[id_]
    mention_scores = [process.extractOne(i.text, syns, scorer=QRatio)[1] for i in mentions]
    spans = []
    for mention, score in zip(mentions, mention_scores):
        if score == max(mention_scores) and score > 70:
            spans.append((mention.start_pos, mention.end_pos))
    return spans


def find_exact_mentions(substrings, string):
    mentions = []
    for substring in substrings:
        if len(substring) < 2:
            continue
        for m in re.finditer(re.escape(substring.lower()), string.lower()):
            mentions.append(m.span())

    return mentions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bel", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    cuid_to_synonyms = defaultdict(set)

    chebi_cuid_to_name = {}
    with open("data/chebi_compounds.tsv") as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        for row in reader:
            chebi_cuid_to_name[row["ID"]] = row["NAME"]
            cuid_to_synonyms[row["NAME"]].add(row["NAME"])
    with open("data/chebi_synonyms.tsv") as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        for row in reader:
            name = chebi_cuid_to_name[row["COMPOUND_ID"]]
            cuid_to_synonyms["CHEBI:" + name].add(row["NAME"])

    with open("data/hgnc.txt") as f:
        hgnc_synonyms = {}
        reader = csv.DictReader(f, dialect="excel-tab")
        for row in reader:
            synonyms = set()
            for k, v in row.items():
                if "HGNC ID" not in k:
                    if "," in v:
                        synonyms.update(v.split(","))
                    else:
                        synonyms.add(v)
            synonyms = sorted(i.strip().strip("\'\"") for i in synonyms if i)
            id_ = "HGNC:"+row["Approved symbol"]
            cuid_to_synonyms[id_] = synonyms


    with open("data/mgi.txt") as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        for row in reader:
            synonyms = set()
            try:
                synonyms.update(row["Marker Synonyms (pipe-separated)"].split("|"))
                synonyms.add(row["Marker Symbol"])
                synonyms.add(row["Marker Name"])
                cuid_to_synonyms["MGI:" + row["Marker Symbol"]] = synonyms
            except AttributeError:
                continue

    with open("data/gobp_dict.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids, syn = line.split("||")
            for id_ in ids.split("|"):
                cuid = "GOBP:" + id_
                cuid_to_synonyms[cuid].add(syn)

    with open("data/CTD_diseases.csv") as f:
        reader = csv.DictReader(f)
        for line in reader:
            if "MESH" in line["DiseaseID"]:
                cuid = "MESHD:" + line["DiseaseName"]
                synonyms = set(line["Synonyms"].split("|"))
                synonyms.add(line["DiseaseName"])
                cuid_to_synonyms[cuid] = synonyms

    with open("data/gene_info") as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        for line in reader:
            synonyms = set(line["Synonyms"].split("|"))
            synonyms.add(line["Symbol"])
            cuid = "EGID:" + line["GeneID"]
            cuid_to_synonyms[cuid] = synonyms

    embeddings = sent2vec.Sent2vecModel()
    # embeddings.load_model("/vol/fob-wbia-vol2/wbi/resources/embeddings/BioSentVec_PubMed_MIMICIII-bigram_d700.bin")
    embeddings.load_model("/vol/fob-wbia-vol2/wbi/resources/embeddings/wiki_unigrams.bin")
    tagger = MultiTagger.load(["hunflair-gene", "hunflair-chemical", "hunflair-disease"])

    sentence_splitter = SciSpacySentenceSplitter()
    tree = etree.parse(args.bel)
    for document in tqdm(list(tree.xpath('//document'))):
        for passage in document.xpath("./passage"):
            text = passage.xpath("./text")[0].text
            for ann in passage.xpath("./annotation"):
                cuid = None
                for infon in ann.xpath("./infon"):
                    if infon.attrib["key"] in {"HGNC", "MGI", "CHEBI", "GOBP", "MESHD", "EGID"}:
                        cuid = infon.attrib["key"] + ":" + infon.text
                if cuid:
                    if cuid not in cuid_to_synonyms:
                        document.remove(passage)
                        break

                    # exact_mentions = find_exact_mentions(cuid_to_synonyms[cuid], text)
                    # if exact_mentions:
                    #     mentions = exact_mentions
                    # else:
                    if cuid.startswith("GOBP:"):
                        mentions = [find_matches_sent2vec(cuid_to_synonyms[cuid], text, embeddings)[0]]
                    else:
                        mentions = get_mentions_hunflair(cuid, text, cuid_to_synonyms, tagger, sentence_splitter)
                    if mentions:
                        mention = mentions[0]
                        mention_text = text[mention[0]:mention[1]]
                        ann.xpath("./text")[0].text = mention_text
                        for loc in ann.xpath("./location"):
                            ann.remove(loc)
                        for m in re.finditer(re.escape(mention_text), text):
                            ann.append(etree.Element("location", start=str(m.span()[0]), offset=str(m.span()[1]-m.span()[0])))
                    else:
                        document.remove(passage)
                        break
    with open(args.out, "wb") as f:
        tree.write(f)
