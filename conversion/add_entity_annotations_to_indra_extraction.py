import sent2vec
from bioc import BioCJSONReader, BioCCollection, BioCDocument, BioCAnnotation, BioCPassage
from indra.statements import Statement
import csv

from conversion.utils import find_matches_sent2vec



reader = BioCJSONReader('foo.json')
reader.read()


n_entities = 0
for document in reader.collection.documents:
    for passage in document.passages:
        stmts = []
        for stmt_json in passage.infons["indra"]:
            stmts.append(Statement._from_json(stmt_json))
        for ann in passage.annotations:
            ann.id = str(ann.id)
            print(ann)






