import spacy
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.cli import download

# --- KONFIGURACJA ---
label_map = {
    "BRAK_ZDARZENIA": 0,
    "PRZESTEPSTWO": 1,
    "POLITYKA": 2,
    "BIZNES": 3,
    "KATASTROFA": 4,
    "WYPADEK": 5
}

# --- ŁADOWANIE DANYCH ---
try:
    with open('../data/train_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Nie znaleziono pliku 'train_dataset.json'.")
    exit()

df = pd.DataFrame(data)

df['label_id'] = df['Etykieta'].map(label_map)
df = df.dropna(subset=['label_id'])
df['label_id'] = df['label_id'].astype(int)

X = df['Zdanie'].values
y = df['label_id'].values

print(f"Wczytano poprawnie: {len(X)} rekordów.")

# --- PODZIAŁ DANYCH (TRENING / WALIDACJA / TEST) ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.1, 
    stratify=y,     
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5, 
    stratify=y_temp, 
    random_state=42
)

print(f"\n--- WYNIK PODZIAŁU ---")
print(f"Zbiór Treningowy (do nauki):    {len(X_train)} rekordów")
print(f"Zbiór Walidacyjny (do tuningu): {len(X_val)} rekordów")
print(f"Zbiór Testowy:      {len(X_test)} rekordów")

unique, counts = np.unique(y_test, return_counts=True)
print("\nRozkład klas w zbiorze testowym (450):")

inv_map = {v: k for k, v in label_map.items()}
for label_id, count in zip(unique, counts):
    print(f"  {inv_map[label_id]}: {count}")


# --- PRZYGOTOWANIE SPACY ---
print("\n--- KONWERSJA DANYCH DO FORMATU SPACY ---")
try:
    nlp = spacy.load("pl_core_news_lg")
except OSError:
    print("Brak modelu 'pl_core_news_lg'. Pobieram...")
    download("pl_core_news_lg")
    nlp = spacy.load("pl_core_news_lg")

# --- FUNKCJA KONWERTUJĄCA ---
def save_spacy_data(texts, labels, output_file, oversample=False):
    db = DocBin()
    
    df_temp = pd.DataFrame({'text': texts, 'label_id': labels})
    cats_list = ["BRAK_ZDARZENIA", "PRZESTEPSTWO", "POLITYKA", "BIZNES", "KATASTROFA", "WYPADEK"]
    
    # --- OVERSAMPLING ---
    if oversample:
        print(f"Oversampling włączony dla {output_file}...")
        max_size = df_temp['label_id'].value_counts().max()
        
        new_df_parts = []
        for class_id, group in df_temp.groupby('label_id'):
            if len(group) < max_size:
                resampled = group.sample(max_size, replace=True, random_state=42)
                new_df_parts.append(resampled)
            else:
                new_df_parts.append(group)
        
        df_temp = pd.concat(new_df_parts).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Liczebność po oversamplingu: {len(df_temp)} (było: {len(texts)})")

    # --- TWORZENIE DOKUMENTÓW SPACY ---
    for text, label_id in tqdm(zip(df_temp['text'], df_temp['label_id']), total=len(df_temp), desc=f"Generowanie {output_file}"):
        doc = nlp.make_doc(str(text))
        
        cats = {category: 0.0 for category in cats_list}
        true_label_name = cats_list[int(label_id)]
        cats[true_label_name] = 1.0
        
        doc.cats = cats
        db.add(doc)
    
    db.to_disk(output_file)
    print(f"Zapisano: {output_file}")

# --- GENEROWANIE PLIKÓW ---
save_spacy_data(X_train, y_train, "../data/data_train.spacy", oversample=True)
save_spacy_data(X_val, y_val, "../data/data_dev.spacy", oversample=False)
save_spacy_data(X_test, y_test, "../data/data_test.spacy", oversample=False)

print("\nPliki .spacy gotowe.")