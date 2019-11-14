import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from transformers import BertTokenizer

from .run_bionlp import InputExample, convert_examples_to_features, BertForSequenceMultilabelClassification, BioNLPProcessor
import torch


def predict(pair, pair_data, model, tokenizer):
    examples = []
    for i, mention in enumerate(pair_data['mentions']):
        examples.append(
            InputExample(guid=pair + str(i), text_a=mention[0])
        )
    features = convert_examples_to_features(examples, tokenizer=tokenizer, max_length=256)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
    scores = torch.sigmoid(logits).detach().cpu().tolist()
    label_list = BioNLPProcessor.get_labels()

    label_scores = defaultdict(float)
    alphas = []
    for score in scores:
        assert len(score) == len(label_list)
        for s, label in zip(score, label_list):
            label_scores[label] = max(s, label_scores[label])
        alphas.append(max(score))

    return {
        "entities": pair.split(','),
        "labels": [[l, label_scores[l]] for l in label_list],
        "mentions": pair_data["mentions"],
        "alphas": alphas
    }





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model")

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                              do_lower_case=True)
    model = BertForSequenceMultilabelClassification.from_pretrained(args.model_name_or_path)

    with args.input.open() as f_in, args.output.open('w') as f_out:
        data = json.load(f_in)
        for pair in tqdm(data):
            preds = predict(pair, data[pair], model=model, tokenizer=tokenizer)
            f_out.write(json.dumps(preds) + "\n")





