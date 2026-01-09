'''
Docstring for src.train_auditor


Loads auditor dataset
Splits into train and test sets

'''

import json
from sklearn.feature_extraction.text import TfidfVectorizer


train_rows = []
test_rows = []

def load_auditor_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            split = data.get("split")
            if split == "train":
                train_rows.append(data)
            elif split == "test":
                test_rows.append(data)

    return train_rows, test_rows


def main():
    file_path = "data/processed/auditor_dataset.jsonl"
    train_data, test_data = load_auditor_dataset(file_path)

    print(f"Loaded {len(train_data)} training samples.")
    print(f"Loaded {len(test_data)} testing samples.")
    
    x_train_text = [row["output_text"] for row in train_data]
    y_train = [row["y"] for row in train_data] #0(non-harmful), 1(harmful)
    x_test_text = [row["output_text"] for row in test_data]
    y_test = [row["y"] for row in test_data]

    print("Train positive rate :", sum(y_train)/len(y_train))
    print("Test positive rate :", sum(y_test)/len(y_test))

    # Train positive rate = .33
    # Test positive rate = 1 (all harmful)
 
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2) #Defines a feature space
    # ^ represents each output using words and two-word phrases that appear in at least two training outputs capped at 5000 features

    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    print("X_train shape:", x_train.shape)
    print("X_test shape:", x_test.shape)
    
    #Prints     (30, 40)
    #           (5, 40)

    '''
    While I wanted to train the auditor to distinguish refusal behavior from
    non-refusal behavior, the base model (Flan-T5) did not refuse any.

    Because the base model never refused harmful prompts 
    in the sampled data, the auditor is trained only on 
    benign vs harmful-topic outputs, not on refusal behavior.
    As a result, the auditor’s success reflects topic 
    generalization rather than compliance-versus-refusal 
    discrimination.

    This is where I had to pivot from my original plan.

    '''







    

if __name__ == "__main__":
    main()
