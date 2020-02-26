bionlp_classification: embeddings/data/BioNLP-ST_2013_PC_train.csv \
				       embeddings/data/BioNLP-ST_2013_PC_dev.csv \
                       embeddings/data/BioNLP-ST_2013_GE_train.csv \
				       embeddings/data/BioNLP-ST_2013_GE_dev.csv \
					   embeddings/data/BioNLP-ST_2011_Epi_train.csv \
				       embeddings/data/BioNLP-ST_2011_Epi_dev.csv \
					   embeddings/data/BioNLP-ST_2011_GE_train.csv \
				       embeddings/data/BioNLP-ST_2011_GE_dev.csv \
					   embeddings/data/BioNLP-ST_2013_train.csv \
				       embeddings/data/BioNLP-ST_2013_dev.csv \
					   embeddings/data/BioNLP-ST_2011_train.csv \
				       embeddings/data/BioNLP-ST_2011_dev.csv \
					   embeddings/data/BioNLP-ST_train.csv \
				       embeddings/data/BioNLP-ST_dev.csv

embeddings/data/BioNLP-ST_2013_PC_train.csv:
	python -m conversion.standoff_to_text_classification data/BioNLP-ST_2013_PC_training_data embeddings/data/BioNLP-ST_2013_PC_train.csv
embeddings/data/BioNLP-ST_2013_PC_dev.csv:
	python -m conversion.standoff_to_text_classification data/BioNLP-ST_2013_PC_development_data embeddings/data/BioNLP-ST_2013_PC_dev.csv

embeddings/data/BioNLP-ST_2013_GE_train.csv:
	python -m conversion.standoff_to_text_classification data/BioNLP-ST-2013_GE_train_data_rev3 embeddings/data/BioNLP-ST_2013_GE_train.csv
embeddings/data/BioNLP-ST_2013_GE_dev.csv:
	python -m conversion.standoff_to_text_classification data/BioNLP-ST-2013_GE_devel_data_rev3 embeddings/data/BioNLP-ST_2013_GE_dev.csv

embeddings/data/BioNLP-ST_2011_Epi_train.csv:
	python -m conversion.standoff_to_text_classification data/BioNLP-ST_2011_Epi_and_PTM_training_data_rev1 embeddings/data/BioNLP-ST_2011_Epi_train.csv
embeddings/data/BioNLP-ST_2011_Epi_dev.csv:
	python -m conversion.standoff_to_text_classification data/BioNLP-ST_2011_Epi_and_PTM_development_data_rev1 embeddings/data/BioNLP-ST_2011_Epi_dev.csv

embeddings/data/BioNLP-ST_2011_GE_train.csv:
	python -m conversion.standoff_to_text_classification data/BioNLP-ST_2011_genia_train_data_rev1 embeddings/data/BioNLP-ST_2011_GE_train.csv
embeddings/data/BioNLP-ST_2011_GE_dev.csv:
	python -m conversion.standoff_to_text_classification data/BioNLP-ST_2011_genia_devel_data_rev1 embeddings/data/BioNLP-ST_2011_GE_dev.csv

embeddings/data/BioNLP-ST_2013_train.csv:
	tail -n+2 embeddings/data/BioNLP-ST_2013_PC_train.csv | cat embeddings/data/BioNLP-ST_2013_GE_train.csv - > embeddings/data/BioNLP-ST_2013_train.csv
embeddings/data/BioNLP-ST_2013_dev.csv:
	tail -n+2 embeddings/data/BioNLP-ST_2013_PC_dev.csv | cat embeddings/data/BioNLP-ST_2013_GE_dev.csv - > embeddings/data/BioNLP-ST_2013_dev.csv

embeddings/data/BioNLP-ST_2011_train.csv:
	tail -n+2 embeddings/data/BioNLP-ST_2011_GE_train.csv | cat embeddings/data/BioNLP-ST_2011_Epi_train.csv - > embeddings/data/BioNLP-ST_2011_train.csv
embeddings/data/BioNLP-ST_2011_dev.csv:
	tail -n+2 embeddings/data/BioNLP-ST_2011_GE_dev.csv | cat embeddings/data/BioNLP-ST_2011_Epi_dev.csv - > embeddings/data/BioNLP-ST_2011_dev.csv

embeddings/data/BioNLP-ST_train.csv:
	tail -n+2 embeddings/data/BioNLP-ST_2011_train.csv | cat embeddings/data/BioNLP-ST_2013_train.csv - > embeddings/data/BioNLP-ST_train.csv
embeddings/data/BioNLP-ST_dev.csv:
	tail -n+2 embeddings/data/BioNLP-ST_2011_dev.csv | cat embeddings/data/BioNLP-ST_2013_dev.csv - > embeddings/data/BioNLP-ST_dev.csv
