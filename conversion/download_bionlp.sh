#!/usr/bin/env bash

cd data
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_train_data_rev1.tar.gz'
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_devel_data_rev1.tar.gz'
tar xf BioNLP-ST_2011_genia_train_data_rev1.tar.gz
tar xf BioNLP-ST_2011_genia_devel_data_rev1.tar.gz

wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Epi_and_PTM_training_data_rev1.tar.gz'
wget 'http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Epi_and_PTM_development_data_rev1.tar.gz'
tar xf BioNLP-ST_2011_Epi_and_PTM_training_data_rev1.tar.gz
tar xf BioNLP-ST_2011_Epi_and_PTM_development_data_rev1.tar.gz

wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_train_data_rev3.tar.gz?attredirects=0'
wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_devel_data_rev3.tar.gz?attredirects=0'
tar xf BioNLP-ST-2013_GE_train_data_rev3.tar.gz?attredirects=0
tar xf BioNLP-ST-2013_GE_devel_data_rev3.tar.gz?attredirects=0

wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_training_data.tar.gz?attredirects=0'
wget 'http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_development_data.tar.gz?attredirects=0'
tar xf 'BioNLP-ST_2013_PC_training_data.tar.gz?attredirects=0'
tar xf 'BioNLP-ST_2013_PC_development_data.tar.gz?attredirects=0'

rm -f BioNLP*.tar.gz*
