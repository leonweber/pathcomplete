import os
import sys
from collections import defaultdict

import re
from operator import itemgetter

import requests
from flair.tokenization import SegtokTokenizer
from networkx.utils import UnionFind
from sklearn.metrics.pairwise import cosine_similarity

def overlaps(a, b):
    a = [int(i) for i in a]
    b = [int(i) for i in b]
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def geneid_to_uniprot(symbol, mg):
    try:
        with HiddenPrints():
            res = mg.getgene(str(symbol), size=1, fields='uniprot', species='human')
    except requests.exceptions.HTTPError:
        print("Couldn't find %s" % symbol)
        return None
    if res and 'uniprot' in res:
        if 'Swiss-Prot' in res['uniprot']:
            uniprot = res['uniprot']['Swiss-Prot']
            if isinstance(uniprot, list):
                return uniprot
            else:
                return [uniprot]

    print("Couldn't find %s" % symbol)
    return None


def hgnc_to_uniprot(symbol, mapping, mg):
    try:
        symbol = mapping[symbol]
        return symbol
    except KeyError as ke:
        with HiddenPrints():
            res = mg.query('symbol:%s' % symbol, size=1, fields='uniprot')['hits']
        if res and 'uniprot' in res[0]:
            if 'Swiss-Prot' in res[0]['uniprot']:
                uniprot = res[0]['uniprot']['Swiss-Prot']
                return [uniprot]

        print("Couldn't find %s" % symbol)
        return []


def natural_language_to_uniprot(string, mg):
    string = string.replace("/", " ").replace("+", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    with HiddenPrints():
        res = mg.query(string, size=1, fields='uniprot', species='human')['hits']
    if res and 'uniprot' in res[0]:
        if 'Swiss-Prot' in res[0]['uniprot']:
            uniprot = res[0]['uniprot']['Swiss-Prot']
            return uniprot

    return None


def get_pfam(uniprot, mg):
    with HiddenPrints():
        res = mg.query('uniprot:'+uniprot, size=1, fields='pfam')['hits']


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


datadir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data')
TAX_IDS = {
    'human': '9606',
    'rat': '10116',
    'mouse': '10090',
    'rabbit': '9986',
    'hamster': '10030'
}


def convert_genes(genes, mapping):
    converted_genes = set()

    for gene in genes:
        if gene in mapping:
            converted_genes.update(mapping[gene])

    return converted_genes


def load_homologene_uf(species=None, gene_conversion=None):
    if not species:
        species = set()

    prev_cluster_id = None
    cluster = set()
    uf = UnionFind()
    with open(os.path.join(datadir, "homologene.data")) as f:
        for line in f:
            line = line.strip()
            cluster_id, tax_id, gene_id = line.split('\t')[:3]
            if gene_id in gene_conversion and tax_id in species:
                cluster.update(gene_conversion[gene_id])
            if prev_cluster_id and cluster_id != prev_cluster_id:
                if cluster:
                    uf.union(*cluster)
                cluster = set()

            prev_cluster_id = cluster_id

    return uf


def load_homologene(species=None, gene_conversion=None):
    if not species:
        species = set()

    gene_mapping = defaultdict(set)
    prev_cluster_id = None
    human_genes = set()
    other_genes = set()
    with open(os.path.join(datadir, "homologene.data")) as f:
        for line in f:
            line = line.strip()
            cluster_id, tax_id, gene_id = line.split('\t')[:3]

            if prev_cluster_id and cluster_id != prev_cluster_id:
                if gene_conversion:
                    other_genes = convert_genes(other_genes, gene_conversion)
                    human_genes = convert_genes(human_genes, gene_conversion)

                for other_gene in other_genes:
                    gene_mapping[other_gene].update(human_genes)

                human_genes = set()
                other_genes = set()

            if tax_id == '9606':
                human_genes.add(gene_id)
            if tax_id in species:
                other_genes.add(gene_id)

            prev_cluster_id = cluster_id

    for other_gene in other_genes:
        gene_mapping[other_gene].update(human_genes)

    return gene_mapping

def slugify(value):
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)

    return value


def find_matches_sent2vec(substrings, string, embeddings, threshold=None):
    tokenizer = SegtokTokenizer()
    string_tokens = tokenizer.tokenize(string.lower().replace("-", " "))
    found_spans = []
    matches = []
    for substring in substrings:
        substring = substring.lower().replace("-", " ")
        substring_tok = " ".join(t.text for t in tokenizer.tokenize(substring))
        substring_emb = embeddings.embed_sentence(substring_tok)
        for ws in range(1, len(substring_tok.split())+3):
            for i in range(len(string_tokens) - ws + 1):
                start = i
                end = i +ws -1
                span = (string_tokens[start].start_pos, string_tokens[end].end_pos)

                # if any(overlaps(span, s) for s in found_spans):
                #     continue

                window = " ".join(t.text for t in string_tokens[i:i+ws])
                window_emb = embeddings.embed_sentence(window)
                dist = 1 - cosine_similarity(substring_emb, window_emb).squeeze()

                if threshold is None or dist < threshold:
                    matches.append((span, dist))
                    found_spans.append(span)

    matches = sorted(matches, key=itemgetter(1))

    return [i[0] for i in matches]

