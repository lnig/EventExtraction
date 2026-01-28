import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --- KONFIGURACJA ---
label_map = {
    "BRAK_ZDARZENIA": 0,
    "PRZESTEPSTWO": 1,
    "POLITYKA": 2,
    "BIZNES": 3,
    "KATASTROFA": 4,
    "WYPADEK": 5
}

# --- WCZYTYWANIE DANYCH ---
print("Wczytywanie i przygotowanie danych...")
try:
    with open('../data/train_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Brak pliku train_dataset.json")
    exit()

df = pd.DataFrame(data)
df['label_id'] = df['Etykieta'].map(label_map)
df = df.dropna(subset=['label_id'])
df['label_id'] = df['label_id'].astype(int)

X = df['Zdanie'].values
y = df['label_id'].values

# --- PODZIAŁ ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# --- WEKTORYZACJA ---
print("Zamiana tekstu na liczby")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) 

# --- DEFINICJA MODELI KLASYCZNYCH ---
models = {
    "7. Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "8. SVM (Linear)": SVC(kernel='linear', class_weight='balanced'),
    "9. Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

results = []

# --- TRENOWANIE ---
print("\n--- ROZPOCZYNAM TRENING MODELI KLASYCZNYCH ---")
for name, model in models.items():
    print(f"Trenowanie: {name}...")
    
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy": accuracy
    })

# --- WYŚWIETLANIE WYNIKÓW ---
df_classic = pd.DataFrame(results)
df_classic = df_classic.sort_values(by="F1 Score", ascending=False)

df_display = df_classic.copy()
for col in ["Precision", "Recall", "F1 Score", "Accuracy"]:
    df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")

print("\n" + "="*60)
print("WYNIKI ML (Klasyczne)")
print("="*60)
try:
    import tabulate
    print(df_display.to_markdown(index=False))
except ImportError:
    print(df_display.to_string(index=False))