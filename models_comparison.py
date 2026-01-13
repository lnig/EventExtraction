import spacy
import pandas as pd
from spacy.tokens import DocBin
from spacy.training import Example
import tabulate
import os

models_dirs = {
    "1. Baseline": "../models/output_ensemble/model-best",     
    "2. BOW (Simple)": "../models/output_bow/model-best",       
    "3. Dropout (Tuned)": "../models/output_dropout/model-best",
    "4. Bigram (Context)": "../models/output_bigram/model-best", 
    "5. Light (Fast)": "../models/output_light/model-best",
    "6. Herbert": "../models/output_herbert/model-best",
}

print("Wczytywanie pliku test.spacy...")
if not os.path.exists("../data/data_test.spacy"):
    print("Brak pliku test.spacy")
    exit()

doc_bin = DocBin().from_disk("../data/data_test.spacy")
results = []

print("\n--- PORÓWNANIE MODELI ---")

for name, model_path in models_dirs.items():
    try:
        if not os.path.exists(model_path):
            print(f"Pominięto (brak folderu): {model_path}")
            continue
            
        nlp = spacy.load(model_path)
        docs_test = list(doc_bin.get_docs(nlp.vocab))
        
        examples = []
        for doc in docs_test:
            pred_doc = nlp(doc.text)
            examples.append(Example(pred_doc, doc))
            
        scores = nlp.evaluate(examples)
        
        results.append({
            "Model": name,
            "Precision": scores.get("cats_macro_p", 0.0),
            "Recall": scores.get("cats_macro_r", 0.0),
            "F1 Score": scores.get("cats_macro_f", 0.0),
            "Accuracy": scores.get("cats_score", 0.0)
        })
        
    except Exception as e:
        print(f"⚠️ Błąd przy modelu {name}: {e}")

df = pd.DataFrame(results)

if not df.empty:
    df = df.sort_values(by="F1 Score", ascending=False)
    
    df_display = df.copy()
    cols = ["Precision", "Recall", "F1 Score", "Accuracy"]
    for col in cols:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")
    
    try:
        print(df_display.to_markdown(index=False))
    except ImportError:
        print(df_display.to_string(index=False))