#!/usr/bin/env bash

#wget https://stringdb-static.org/download/protein.links.v11.0/9606.protein.links.v11.0.txt.gz \
# && gunzip 9606.protein.links.v11.0.txt.gz && mv 9606.protein.links.v11.0.txt data/
#wget https://string-db.org/mapping_files/uniprot/human.uniprot_2_string.2018.tsv.gz && \
# gunzip human.uniprot_2_string.2018.tsv.gz && mv human.uniprot_2_string.2018.tsv data/
python -m conversion.string_to_uniprot --links data/9606.protein.links.v11.0.txt \
 --mapping data/human.uniprot_2_string.2018.tsv --out data/string_v11_human_highconf.tsv



