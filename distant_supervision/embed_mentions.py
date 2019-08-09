from bert_serving.client import BertClient
from pytorch_transformers import BertTokenizer
import argparse
import json
import h5py

from tqdm import tqdm

import numpy as np



def get_proteins(data):
    proteins = set()
    with open(data) as f:
        data = json.load(f)
        for k in data:
            p1, _, p2 = k.split(',')
            proteins.add(p1)
            proteins.add(p2)
    
    return proteins


def truncate(text, size, center):
    start = max(0, min(center - size//2, len(text)-size))
    end = min(start+size, len(text))
    if end == 0:
        end = None
    text = text[start:end]


    return text, start


def embed(mentions, proteins, fname, tokenizer):
    f_out = h5py.File(fname, 'w')
    bc = BertClient()
    chunk_texts = []
    chunk_pmids = []
    chunk_ent_spans = []
    chunk_doc_spans = []
    prev_protein_id = None
    embeddings = f_out.create_group("embeddings")
    pmids = f_out.create_group("pmids")
    doc_spans = f_out.create_group("doc_spans")
    pbar = tqdm(total=len(proteins))
    for mention in mentions:
        fields = mention.split('\t')
        protein_id = fields[0]
        if protein_id not in proteins:
            continue

        if prev_protein_id and protein_id != prev_protein_id:
            embs, tokens = bc.encode(chunk_texts, show_tokens=True, is_tokenized=True)

            entity_embs = []
            for i, (start, end) in enumerate(chunk_ent_spans):
                emb = embs[i, start:end, :].mean(axis=0)
                entity_embs.append(emb)

            embeddings.create_dataset(prev_protein_id, data=np.array(entity_embs))
            pmids.create_dataset(prev_protein_id, data=np.array(chunk_pmids, dtype="int32"))
            doc_spans.create_dataset(prev_protein_id, data=np.array(chunk_doc_spans, dtype="int32"))

            chunk_texts = []
            chunk_pmids = []
            chunk_ent_spans = []
            chunk_doc_spans = []
            pbar.update(1)

            del entity_embs
            del embs
            del tokens
            

        ent_span = (int(fields[4]), int(fields[5]))
        tokens = []

        for i, token in enumerate(fields[2].split()):
            subtokens = tokenizer.tokenize(token)

            if i == ent_span[0]:
                new_begin = len(tokens)

            if i == ent_span[1]:
                new_end = len(tokens)

            tokens.extend(subtokens)


        tokens, shift = truncate(tokens, size=510, center=new_begin)
        chunk_texts.append(tokens)
        chunk_pmids.append(fields[3])
        chunk_ent_spans.append( (new_begin-shift, new_end-shift) )
        chunk_doc_spans.append( (int(fields[6]), int(fields[7])) )
        
        prev_protein_id = protein_id
    
    f_out.close()
    pbar.close()

       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mentions')
    parser.add_argument('--data')
    parser.add_argument('--bert')

    args = parser.parse_args()

    # args.mentions = "/mnt/fob-wbia-vol2/wbi/weberple/mentions.txt"
    # args.data = "data/PathwayCommons11.reactome.hgnc.txt_small.json"
    # args.bert = '/home/weberple/biobert_v1.1_pubmed'

    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=False)

    proteins = get_proteins(args.data)
    with open(args.mentions) as f:
        embed(f, proteins, 'embeddings.hdf5', tokenizer)

    