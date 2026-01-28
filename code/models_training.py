import os
from spacy.cli.train import train

# --- LISTA EKSPERYMENTÓW ---
experiments = [
    {
        "name": "Exp_A_Ensemble",
        "config": "../config/config.cfg",      
        "output": "../models/output_ensemble"
    },
    {
        "name": "Exp_B_BOW",
        "config": "../config/config_bow.cfg", 
        "output": "../models/output_bow"
    },
    {
        "name": "Exp_C_Dropout",
        "config": "../config/config_dropout.cfg",
        "output": "../models/output_dropout"
    },
    {
        "name": "Exp_D_Bigram",
        "config": "../config/config_bigram.cfg", 
        "output": "../models/output_bigram"
    },
    {
        "name": "Exp_E_Light",
        "config": "../config/config_light.cfg", 
        "output": "../models/output_light"
    }
]

# --- PĘTLA TRENINGOWA ---
for exp in experiments:
    print(f"\nTrenowanie: {exp['name']}...")
    
    if not os.path.exists(exp['config']):
        print(f"Brak pliku {exp['config']}!.")
        continue

    # --- URUCHOMIENIE TRENINGU ---
    try:
        train(
            exp['config'],
            exp['output'],
            overrides={
                "paths.train": "../data/data_train.spacy",
                "paths.dev": "../data/data_dev.spacy"
            },
            use_gpu=-1
        )
        print(f"Zakończono: {exp['name']}")
    except Exception as e:
        print(f"Błąd podczas treningu {exp['name']}: {e}")