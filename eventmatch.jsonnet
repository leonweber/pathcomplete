{
  "dataset_reader": {
    "type": "streader",
    "lazy": "false",
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "do_lowercase": true,
        "use_starting_offsets": true
      }
    }
  },
  "train_data_path": "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_training_data.tar.gz",
  "validation_data_path": "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_development_data.tar.gz",
  "model": {
    "type": "eventmatch",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "top_layer_only": true,
        "requires_grad": true
      }
    },
    "encoder": {
      "type": "bertpooler",
      "bert_dim": 768
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 8
  },
  "trainer": {
    "optimizer": {
      "type": "bert_adam",
      "lr": 3e-5
    },
    "validation_metric": "+f1",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 100,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}
