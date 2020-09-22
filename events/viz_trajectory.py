import argparse
import json
import shutil
from pathlib import Path
import os
import logging

from events.model import EventExtractor
from events.parse_standoff import StandoffAnnotation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path)
    parser.add_argument("--dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    args = parser.parse_args()

    with args.config.open() as f:
        config = json.load(f)
    config["output_dir"] = Path("events/foo")
    config["small"] = True
    model = EventExtractor.load_from_checkpoint(args.model, config=config).to("cuda")
    with args.file.with_suffix(".txt").open() as f:
        text = f.read()
    with args.file.with_suffix(".a1").open() as f:
        a1_lines = f.readlines()
    a2_lines = []
    ann = StandoffAnnotation(a1_lines, a2_lines)
    result, batches = model.predict(text=text, ann=ann, fname=args.file.name,
                                    return_batches=True)
    batches.append({"a2_lines": result})
    result_dir = args.dir/args.file.name
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    last_a2_lines = None
    i = 1
    for batch in batches:
        result_file = result_dir/(str(i).zfill(4))
        try:
            print(model.tokenizer.decode(batch["input_ids"][0].tolist(), skip_special_tokens=True))
            print("Edge:", batch["pred_edge"])
            print("Trigger:", batch["pred_trigger"])
            print()
        except KeyError:
            continue

        if "a2_lines" in batch and batch["a2_lines"] != last_a2_lines:
            with result_file.with_suffix(".txt").open("w") as f:
                f.write(text)
            with result_file.with_suffix(".a1").open("w") as f:
                f.writelines(a1_lines)
            with result_file.with_suffix(".a2").open("w") as f:
                f.write(batch["a2_lines"])

            i += 1

        if "a2_lines" in batch:
            last_a2_lines = batch["a2_lines"]
        else:
            last_a2_lines = None




