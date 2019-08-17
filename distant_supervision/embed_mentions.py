from bert_serving.client import BertClient
from pytorch_transformers import BertTokenizer
import argparse
import json
import h5py

from tqdm import tqdm

import numpy as np


MAX_CHUNK_SIZE = 64 * 100



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


def embed(mentions, proteins, h5_file, tokenizer):
    bc = BertClient()
    chunk_texts = []
    chunk_pmids = []
    chunk_ent_spans = []
    chunk_doc_spans = []
    prev_protein_id = None
    embeddings = h5_file.create_group("embeddings")
    pmids = h5_file.create_group("pmids")
    doc_spans = h5_file.create_group("doc_spans")
    with tqdm(total=len(proteins)) as pbar:
        for mention in mentions:
            fields = mention.split('\t')
            protein_id = fields[0]
            if protein_id not in proteins:
                continue

            if prev_protein_id and protein_id != prev_protein_id or len(chunk_texts) > MAX_CHUNK_SIZE:
                # embs  = bc.fetch_all(concat=True, sort=True)
                embs = bc.encode(chunk_texts, is_tokenized=True)

                entity_embs = []
                for i, (start, end) in enumerate(chunk_ent_spans):
                    emb = embs[i, start:end, :].mean(axis=0)
                    entity_embs.append(emb)
                
                if prev_protein_id not in embeddings:
                    embeddings.create_dataset(prev_protein_id, data=np.array(entity_embs), compression='gzip', chunks=True, maxshape=(None,entity_embs[0].shape[0]))
                    pmids.create_dataset(prev_protein_id, data=np.array(chunk_pmids, dtype="int32"), chunks=True, maxshape=(None,))
                    doc_spans.create_dataset(prev_protein_id, data=np.array(chunk_doc_spans, dtype="int32"), chunks=True, maxshape=(None,2))
                else:
                    old_ds_len = embeddings[prev_protein_id].shape[0]
                    new_ds_len = old_ds_len + len(entity_embs)
                    embeddings[prev_protein_id].resize(new_ds_len, axis=0)
                    embeddings[prev_protein_id][old_ds_len:] = entity_embs

                    pmids[prev_protein_id].resize(new_ds_len, axis=0)
                    pmids[prev_protein_id][old_ds_len:] = np.array(chunk_pmids, dtype="int32")
                    
                    doc_spans[prev_protein_id].resize(new_ds_len, axis=0)
                    doc_spans[prev_protein_id][old_ds_len:] = np.array(chunk_doc_spans, dtype="int32")



                chunk_texts = []
                chunk_pmids = []
                chunk_ent_spans = []
                chunk_doc_spans = []

                del entity_embs
                del embs
                del tokens

                if prev_protein_id != protein_id:
                    pbar.update(1)
                

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
            # bc.encode([tokens], show_tokens=False, is_tokenized=True, blocking=False)
            chunk_texts.append(tokens)
            chunk_pmids.append(fields[3])
            chunk_ent_spans.append( (new_begin-shift, new_end-shift) )
            chunk_doc_spans.append( (int(fields[6]), int(fields[7])) )
            
            prev_protein_id = protein_id

        if prev_protein_id not in embeddings:
            embeddings.create_dataset(prev_protein_id, data=np.array(entity_embs), compression='gzip', chunks=True, maxshape=(None,entity_embs[0].shape[0]))
            pmids.create_dataset(prev_protein_id, data=np.array(chunk_pmids, dtype="int32"), chunks=True, maxshape=(None,))
            doc_spans.create_dataset(prev_protein_id, data=np.array(chunk_doc_spans, dtype="int32"), chunks=True, maxshape=(None,2))
        else:
            old_ds_len = embeddings[prev_protein_id].shape[0]
            new_ds_len = old_ds_len + len(entity_embs)
            embeddings[prev_protein_id].resize(new_ds_len, axis=0)
            embeddings[prev_protein_id][old_ds_len:] = entity_embs

            pmids[prev_protein_id].resize(new_ds_len, axis=0)
            pmids[prev_protein_id][old_ds_len:] = np.array(chunk_pmids, dtype="int32")
            
            doc_spans[prev_protein_id].resize(new_ds_len, axis=0)
            doc_spans[prev_protein_id][old_ds_len:] = np.array(chunk_doc_spans, dtype="int32")
    

       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mentions', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--bert', required=True)
    parser.add_argument('--out', required=True)

    args = parser.parse_args()

    # args.mentions = "/mnt/fob-wbia-vol2/wbi/weberple/mentions.txt"
    # args.data = "data/PathwayCommons11.reactome.hgnc.txt_small.json"
    # args.bert = '/home/weberple/biobert_v1.1_pubmed'

    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=False)

    proteins = get_proteins(args.data)
    with open(args.mentions) as f, h5py.File(args.out, 'w') as hf:
        embed(f, proteins, hf, tokenizer)

    