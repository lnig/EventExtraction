import json
import os
import time
import random
import typing_extensions as typing
from collections import Counter
import google.generativeai as genai

# --- KONFIGURACJA ---
API_KEY = ""

FILE_RAW_INPUT = "../data/input_data.json"           
FILE_CLASSIFIED = "../data/classified_data.json"
FILE_FINAL = "../data/final_dataset.json"

BATCH_SIZE = 50
TARGET_SIZE = 9000
NON_EVENT_LABEL = "BRAK_ZDARZENIA"
MODEL_NAME = "gemma-2-27b-it"

class ClassificationResult(typing.TypedDict):
    text: str
    label: str

class BatchResponse(typing.TypedDict):
    results: list[ClassificationResult]

# ---  WCZYTYWANIE JSON ---
def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Brak pliku wejsciowego")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                print("Plik nie jest lista")
                return None
            return data
        except json.JSONDecodeError:
            print("Plik nie jest JSONem")
            return None
        
# --- ZAPISYWANIE JSON ---
def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Zapisano dane")

# --- PRZYGOTOWANIE PROMPTA ---
def classify_batch(model, sentences_list):
    prompt = f"""
    Jesteś analitykiem zdarzeń (Event Extraction). Twoim celem jest wykrycie CZY w zdaniu opisano konkretne wydarzenie fizyczne, czy jest to tylko opis, opinia lub stan rzeczy.
    Twoim zadaniem jest przypisanie JEDNEJ etykiety do każdego nagłówka.

    ZASADY ANALIZY
    1. CZY TO FAKT? Odrzuć opinie, przewidywania przyszłości, plany, zapowiedzi i metafory. Interesują nas tylko zdarzenia, które już miały miejsce lub właśnie trwają.
    2. STRUKTURA: Szukaj schematu: [SPRAWCA] -> [CZYNNOŚĆ] -> [OBIEKT].
    3. DOSŁOWNOŚĆ: "Gospodarka tonie" to metafora (BRAK_ZDARZENIA). "Statek tonie" to fakt (KATASTROFA/WYPADEK).

    Analizuj treść pod kątem występowania konkretnych WYDARZEŃ (akcji).

   KATEGORIE I SŁOWA KLUCZE (TRIGGERY):
    1. PRZESTEPSTWO (Kryminalne, naruszenie prawa)
    - Akcje: aresztować, zatrzymać, pobić, ukraść, zabić...
    2. KATASTROFA (Duża skala zniszczeń lub ofiary)
    - Akcje: wybuchnąć, spłonąć, zawalić się, powódź...
    3. WYPADEK (Lokalne, komunikacyjne, jednostkowe)
    - Akcje: kolizja, potrącić, dachować...
    4. BIZNES (Firmy i gospodarka — tylko KONKRETNE DZIAŁANIA)
    - Akcje: kupić firmę, fuzja, zbankrutować...
    5. POLITYKA (Władza, prawo, działania państw i rządów)
    - Akcje: uchwalić ustawę, zdymisjonować, wygrać wybory...
    6. BRAK_ZDARZENIA (Wszystko inne)
    - opinie, sondaże, zapowiedzi, sport, pogoda.

    --- FORMAT WYJŚCIOWY ---
    Zwróć wynik JAKO CZYSTY JSON.
    Struktura:
    {{
        "results": [
            {{ "text": "oryginalny tekst zdania", "label": "NAZWA_KATEGORII" }},
            ...
        ]
    }}

    Lista zdań do analizy:
    {json.dumps(sentences_list, ensure_ascii=False)}
    """

    # --- WYSŁANIE ZAPYTANIA DO AI ---
    try:
        result = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,      
                max_output_tokens=8192
            )
        )
        
        # --- CZYSZCZENIE I ODCZYT ODPOWIEDZI ---
        clean_text = result.text.strip()

        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        
        clean_text = clean_text.strip()

        parsed = json.loads(clean_text)
        return parsed.get("results", [])
    
    except Exception as e:
        print(f"Błąd API: {e}")
        return [] 

def run_classification():
    print(f"\nRozpoczęcie klasyfikacji danych...")
    
    if len(API_KEY) == 0:
        print("Brak klucza API")
        return

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    raw_data = load_json(FILE_RAW_INPUT)
    if not raw_data: return

    total_items = len(raw_data)
    classified_data = []

    # --- PRZETWARZANIE KAWAŁKAMI ---
    for i in range(0, total_items, BATCH_SIZE):
        batch_objects = raw_data[i : i + BATCH_SIZE]
        batch_texts = [obj.get("Zdanie", "") for obj in batch_objects]
        
        print(f"Przetwarzanie od {i}...")
        
        api_results = classify_batch(model, batch_texts)
        
        for idx, original_obj in enumerate(batch_objects):
            new_obj = original_obj.copy()
            etykieta = "ERROR_API"
            
            if idx < len(api_results):
                res = api_results[idx]
                etykieta = res.get("label", "INNE")
            
            new_obj["Etykieta"] = etykieta
            classified_data.append(new_obj)

        save_json(classified_data, FILE_CLASSIFIED)
        time.sleep(1)

    print(f"\nZakończono klasyfikację")

# --- WYRÓWNYWANIE DANYCH ---
def run_balancing():
    data = load_json(FILE_CLASSIFIED)
    if not data: return

    events = []
    non_events = []

    for item in data:
        if item.get("Etykieta") == NON_EVENT_LABEL:
            non_events.append(item)
        else:
            events.append(item)

    num_events = len(events)
    needed = TARGET_SIZE - num_events
    
    random.seed(42)
    random.shuffle(non_events)
        
    if len(non_events) < needed:
        selected_non_events = non_events
    else:
        selected_non_events = non_events[:needed]
    
    final_dataset = events + selected_non_events

    random.shuffle(final_dataset)
    save_json(final_dataset, FILE_FINAL)
    
    counts = Counter([item["Etykieta"] for item in final_dataset])
    for label, count in counts.most_common():
        print(f"   - {label}: {count}")

def main():
    print("")
    run_classification()
    # run_balancing()

if __name__ == "__main__":
    main()