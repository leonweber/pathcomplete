import csv
from collections import defaultdict

import sent2vec
import torch
from flair.data import Sentence
from flair.tokenization import SciSpacySentenceSplitter
from pybiopax import model_from_owl_file
from bioc import BioCJSONWriter, BioCCollection, BioCDocument, BioCPassage, BioCRelation, \
    BioCAnnotation, BioCLocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import re
from flair.models import MultiTagger
from fuzzywuzzy import process
import yappi

from conversion.indra_biopax_processor import BiopaxProcessor
from conversion.util import find_matches_sent2vec


def deduplicate_statements(stmts):
    new_stmts = []
    processed_hashes = set()
    for stmt in stmts:
        hash = stmt.get_hash(refresh=True)
        if hash in processed_hashes:
            continue
        processed_hashes.add(hash)
        new_stmts.append(stmt)

    return new_stmts

def get_agents_mentions(agents, text, synonyms_by_db, tagger):
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


def filter_agent(agent, ids_to_retain):
    retained_bounds = []
    for bound in agent.bound_conditions:
        if any(tuple(i) in ids_to_retain for i in bound.agent.db_refs.items()):
            retained_bounds.append(bound)

    if not any(tuple(i) in ids_to_retain for i in agent.db_refs.items()):
        if not retained_bounds:
            return None

        new_agent = retained_bounds.pop(0).agent
    else:
        new_agent = agent

    new_agent.bound_conditions = retained_bounds

    return new_agent


def get_site_mentions(site, text):
    mentions = []
    res, loc = site
    if res is None:
        res = ""
    res = res.lower()
    loc = loc.lower()
    text = text.lower()
    matches = re.finditer(f"{res}\s*{loc}", text)
    for match in matches:
        mentions.append(match.span())

    if mentions:
        return mentions

    matches = re.finditer(f"\S\S+{loc}", text)
    for match in matches:
        mentions.append(match.span())

    return mentions



if __name__ == '__main__':
    tagger = MultiTagger.load(["hunflair-gene", "hunflair-chemical"])
    synonyms_by_db = {}
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
            id_ = row["HGNC ID"].split(":")[1]
            hgnc_synonyms[id_] = synonyms

        synonyms_by_db["HGNC"] = hgnc_synonyms

    with open("data/chebi.tsv") as f:
        chebi_synonyms = defaultdict(set)
        reader = csv.DictReader(f, dialect="excel-tab")
        for row in reader:
            chebi_synonyms["CHEBI:" + row["COMPOUND_ID"]].add(row["NAME"])
        synonyms_by_db["CHEBI"] = chebi_synonyms

    model = model_from_owl_file("data/PathwayCommons12.reactome.BIOPAX.owl")
    bpp = BiopaxProcessor(model)
    bpp.process_all()

    text_to_stmt = defaultdict(list)
    for stmt in bpp.statements:
        text = stmt.evidence[0].text
        stmt.evidence = None
        text_to_stmt[text].append(stmt)

    writer = BioCJSONWriter()
    collection = BioCCollection()
    writer.collection = collection
    sentence_splitter = SciSpacySentenceSplitter()

    # yappi.set_clock_type("wall")
    # yappi.start()
    for i, (text, stmts) in tqdm(list(enumerate(text_to_stmt.items()))):
        # if i > 10:
        #     yappi.get_func_stats().print_all()
        #     __import__("sys").exit(0)
        if len(text) > 425 or len(text) < 36:
            continue
        stmts = deduplicate_statements(stmts)
        document = BioCDocument()
        document.id = i

        passage = BioCPassage()
        passage.put_infon("type", "paragraph")
        passage.offset = "0"
        passage.text = text
        document.add_passage(passage)
        added_annotations = set()
        n_entities = 0
        for stmt in stmts:
            id_to_mentions = get_agents_mentions(stmt.agent_list_with_bound_condition_agents(), passage.text, tagger=tagger,
                                                                    synonyms_by_db=synonyms_by_db)
            for (db, id_), mentions in id_to_mentions.items():
                ann = BioCAnnotation()
                ann.id = str(n_entities + 1)
                ann.infons["type"] = db
                ann.infons["id"] = id_
                for start, end in mentions:
                    loc = BioCLocation()
                    loc.offset = start
                    loc.length = end-start
                    ann.locations.append(loc)
                if (db, id_) not in added_annotations:
                    passage.add_annotation(ann)
                    n_entities += 1
                    added_annotations.add((db, id_))

            if hasattr(stmt, "position") and hasattr(stmt, "residue"):
                site = (stmt.residue, stmt.position)
                if stmt.position:
                    site_mentions = get_site_mentions(site, text)
                else:
                    site_mentions = None

                if site_mentions:
                    ann = BioCAnnotation()
                    ann.id = str(n_entities + 1)
                    ann.infons["type"] = "RESIDUE"
                    ann.infons["id"] = "".join(site)
                    for start, end in site_mentions:
                        loc = BioCLocation()
                        loc.offset = start
                        loc.length = end-start
                        ann.locations.append(loc)
                    if (ann.infons["type"], ann.infons["id"]) not in added_annotations:
                        passage.add_annotation(ann)
                        added_annotations.add((ann.infons["type"], ann.infons["id"]))
                        n_entities += 1
                else:
                    stmt.position = None
                    stmt.residue = None

            for agent in stmt.agent_list_with_bound_condition_agents():
                filtered_mods = []
                for mod in agent.mods:
                    site = (mod.residue, mod.position)
                    if mod.position:
                        site_mentions = get_site_mentions(site, text)
                    else:
                        site_mentions = None

                    if site_mentions:
                        ann = BioCAnnotation()
                        ann.id = str(n_entities + 1)
                        ann.infons["type"] = "RESIDUE"
                        ann.infons["id"] = "".join(site)
                        for start, end in site_mentions:
                            loc = BioCLocation()
                            loc.offset = start
                            loc.length = end-start
                            ann.locations.append(loc)
                        if (ann.infons["type"], ann.infons["id"]) not in added_annotations:
                            passage.add_annotation(ann)
                            added_annotations.add((ann.infons["type"], ann.infons["id"]))
                            n_entities += 1
                        filtered_mods.append(mod)
                agent.mods = filtered_mods

            for agent_name in stmt._agent_order:
                agent = getattr(stmt, agent_name)
                if isinstance(agent, list):
                    new_agent = []
                    for a in agent:
                        a = filter_agent(a, ids_to_retain=id_to_mentions.keys())
                        if a:
                            new_agent.append(a)
                else:
                    new_agent = filter_agent(agent, ids_to_retain=id_to_mentions.keys())
                setattr(stmt, agent_name, new_agent)


        passage.put_infon("indra", [])
        for stmt in stmts:
            if None not in stmt.agent_list():
                passage.infons["indra"].append(stmt.to_json())

        if stmts:
            collection.add_document(document)

    with open('foo.json', 'w') as f:
        f.write(writer.tostring())
