'''
Script Responsibilities

1. Load Model + Tokenizer
- Model: google/flan-t5-base
- Tokenizer: matching tokenizer

2. Enumerate Raw Prompt Files
- list all files in data/raw/ ending in .jsonl
- For each file, read line-by-line and parse JSON

3. For each prompt, generate a response
- extract prompt, family, label
- run generation on the prompt
- decode to a string output

4. Write data/processed/outputs.jsonl
Each output should include
- prompt
- family
- label
- model_name
- output_text

{"prompt":"...","family":"encoding","label":1,"model_name":"google/flan-t5-base","output_text":"..."}

Dependencies
- transformers
- torch
- tqdm

'''
import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



MODEL_NAME = "google/flan-t5-base"

jsonfiles = ["data/raw/authority.jsonl", "data/raw/encoding.jsonl", "data/raw/direct_harm.jsonl", "data/raw/benign_prompts.jsonl"]

def main():
    #Load tokenizer
    #Load model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    with open("data/processed/outputs.jsonl", "w", encoding="utf-8") as output_f:
        for file in jsonfiles:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:

                    #Parse JSON line
                    data = json.loads(line.strip())
                    prompt = data.get("prompt")
                    family = data.get("family")
                    label = data.get("label")

                    #Generate output
                    inputs = tokenizer(prompt, return_tensors="pt")
                    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9)
                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    #Takes output and writes to outputs.jsonl
                    output_data = {
                        "prompt": prompt,
                        "family": family,
                        "label": label,
                        "model_name": MODEL_NAME,
                        "output_text": output_text
                    }
                    
                    output_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    


                    





if __name__ == "__main__":
    main()