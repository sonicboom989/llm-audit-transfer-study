

import json
from config import *

input_file = "data/processed/outputs.jsonl"
output_file = "data/processed/auditor_dataset.jsonl"

def main():
    with open(input_file, "r", encoding="utf-8") as input_f, open(output_file, "w", encoding="utf-8") as output_f:

        for line in input_f:
            data = json.loads(line.strip())
            family = data.get("family")
            split = "test" if family == "encoding" else "train"
            label = data.get("label")
            output_text = data.get("output_text")
            y = 0 if label != 1 else (0 if is_refusal(output_text.lower()) else 1)
            
            output_data = {
                "family": family,
                "split": split,
                "y": y,
                "output_text": output_text,
                "prompt_label": label
                    }
            output_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    

if __name__ == "__main__":
    main()