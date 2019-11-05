#!/usr/bin/env bash
# Only run from base dir!

cd data
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_train_data_rev1.tar.gz'
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_devel_data_rev1.tar.gz'
tar xf BioNLP-ST_2011_genia_train_data_rev1.tar.gz
tar xf BioNLP-ST_2011_genia_devel_data_rev1.tar.gz
cd ..
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2011_genia_train_data_rev1/ --out distant_supervision/data/BioNLP-ST_2011_genia/train.json
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2011_genia_devel_data_rev1/ --out distant_supervision/data/BioNLP-ST_2011_genia/dev.json

cd data
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Epi_and_PTM_training_data_rev1.tar.gz'
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Epi_and_PTM_development_data_rev1.tar.gz'
tar xf BioNLP-ST_2011_Epi_and_PTM_training_data_rev1.tar.gz
tar xf BioNLP-ST_2011_Epi_and_PTM_development_data_rev1.tar.gz
cd ..
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2011_Epi_and_PTM_training_data_rev1/ --out distant_supervision/data/BioNLP-ST_2011_epi/train.json
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2011_Epi_and_PTM_development_data_rev1/ --out distant_supervision/data/BioNLP-ST_2011_epi/dev.json

cd data
wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_train_data_rev3.tar.gz?attredirects=0'
wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_devel_data_rev3.tar.gz?attredirects=0'
tar xf BioNLP-ST-2013_GE_train_data_rev3.tar.gz?attredirects=0
tar xf BioNLP-ST-2013_GE_devel_data_rev3.tar.gz?attredirects=0
cd ..
python conversion/standoff_to_ds.py --data data/BioNLP-ST-2013_GE_train_data_rev3/ --out distant_supervision/data/BioNLP-ST_2013_GE/train.json
python conversion/standoff_to_ds.py --data data/BioNLP-ST-2013_GE_devel_data_rev3/ --out distant_supervision/data/BioNLP-ST_2013_GE/dev.json

cd data
wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_training_data.tar.gz?attredirects=0'
wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_development_data.tar.gz?attredirects=0'
tar xf 'BioNLP-ST_2013_PC_training_data.tar.gz?attredirects=0'
tar xf 'BioNLP-ST_2013_PC_development_data.tar.gz?attredirects=0'
cd ..
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2013_PC_training_data/ --out distant_supervision/data/BioNLP-ST_2013_PC/train.json
python conversion/standoff_to_ds.py --data data/BioNLP-ST_2013_PC_development_data/ --out distant_supervision/data/BioNLP-ST_2013_PC/dev.json

cd data
rm -f BioNLP*.tar.gz*

python conversion/combine_bionlp_ds_data.py distant_supervision/data/BioNLP-ST* distant_supervision/data/BioNLP
