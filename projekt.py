import json
import os
import time
import random
import typing_extensions as typing
from collections import Counter
import google.generativeai as genai

API_KEY = ""

FILE_RAW_INPUT = "data/dane_wejsciowe.json"             
FILE_CLASSIFIED = "data/dane_sklasyfikowane.json"
FILE_FINAL = "data/dataset.json"

BATCH_SIZE = 50
TARGET_SIZE = 2000
NON_EVENT_LABEL = "BRAK_ZDARZENIA"
MODEL_NAME = "gemini-2.5-flash"

class ClassificationResult(typing.TypedDict):
    text: str
    label: str

class BatchResponse(typing.TypedDict):
    results: list[ClassificationResult]

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

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Zapisano dane")

def classify_batch(model, sentences_list):
    prompt = f"""
   Jesteś analitykiem zdarzeń (Event Extraction). Twoim celem jest wykrycie CZY w zdaniu opisano konkretne wydarzenie fizyczne, czy jest to tylko opis, opinia lub stan rzeczy.
    Twoim zadaniem jest przypisanie JEDNEJ etykiety do każdego nagłówka.

    ZASADA GLÓWNA:
    Aby zdanie było wydarzeniem, musi spełniać strukturę:
    [KTO/CO (Sprawca)] + [TRIGGER (Czasownik akcji)] + [KOGO/CO (Obiekt/Ofiara)].

    Analizuj treść pod kątem występowania konkretnych WYDARZEŃ (akcji).

    KATEGORIE I SŁOWA KLUCZE (TRIGGERY):
    1. PRZESTEPSTWO (Kryminalne, naruszenie prawa)
    - Akcje: 
        aresztować, zatrzymać, zatrzymanie, ująć, obława, nalot policji, pobić, pobicie, napaść, napad, rozbój, ukraść, kradzież, okraść, rabować,
        włamać się, włamanie, zabić, zamordować, zabójstwo, morderstwo, postrzelić, strzelać, dźgnąć, pchnąć nożem, porwać, uprowadzić, przetrzymywać,
        torturować, zmuszać, handlować ludźmi, przemycać, przemyt, podrabiać, fałszować dokumenty, oszukać, oszustwo, wyłudzić, wyłudzenie,
        defraudować, sprzeniewierzyć, uciec z więzienia, zbiegł z aresztu, poszukiwany listem gończym, zdemolować, demolować, niszczyć mienie, wandalizm,
        podpalić, podpalenie, wtargnąć, napaść na policję, rzucać kamieniami, okupować nielegalnie, użyć przemocy, skazać, wyrok, skazany, oskarżyć, postawić zarzuty,
        akt oskarżenia,ekstradycja, deportować (po przestępstwie)
    - Dotyczy:
        policji, prokuratury, sądów karnych, sprawców, gangów, mafii,
        terrorystów, przemytników.

    2. KATASTROFA (Duża skala zniszczeń lub ofiary)
    - Akcje:
        wybuchnąć, eksplozja, detonacja, spłonąć, pożar, pożar hali, pożar lasu, zawalić się, runąć, katastrofa budowlana, osunięcie ziemi, lawina,
        powódź, fala powodziowa, zalanie miasta, trzęsienie ziemi, tsunami, tornado, huragan, cyklon, burza stulecia, uderzyć żywioł, meteoryt uderzył,
        erupcja wulkanu, skażenie chemiczne, radioaktywne, wyciek gazu, katastrofa ekologiczna, zatrucie wody, ewakuować masowo, ewakuacja tysięcy, 
        rozbił się samolot, wykoleił się pociąg (z ofiarami), statek zatonął

    3. WYPADEK (Lokalne, komunikacyjne, jednostkowe)
    - Akcje:
        wypadek, kolizja, stłuczka, zderzyć się, potrącić, potrącenie, dachować, wpaść do rowu, uderzyć w drzewo, słup, spaść z wysokości,
        utonąć, wypadek przy pracy, przygnieść, porazić prądem, zatrucie czadem, poparzyć się, zasypany w wykopie, zasłabnął za kierownicą

    4. BIZNES (Firmy i gospodarka — tylko KONKRETNE DZIAŁANIA)
    - Akcje:
        kupić firmę, sprzedać spółkę, przejąć spółkę, fuzja, wykupić udziały, wejść na giełdę, IPO, wycofać z giełdy, zbankrutować, upadłość, 
        zlikwidować firmę, zwolnić grupowo, zatrudnić setki, zainwestować, pozyskać inwestora, podpisać kontrakt, restrukturyzacja,zamknąć zakład, 
        zerwać kontrakt, ogłosić wyniki finansowe, rekordowe zyski, ogromna strata, przejąć długi, otworzyć fabrykę, przenieść produkcję, wycofać produkt z rynku

    5. POLITYKA (Władza, prawo, działania państw i rządów)
    - Akcje:
        uchwalić ustawę, podpisać ustawę, zawetować, zmienić prawo, nowelizacja, przyjąć projekt, odrzucić projekt, głosowanie w Sejmie, rozporządzenie, dekret, 
        ogłosić decyzję rządu, mianować,powołać, odwołać ministra, dymisja, podać się do dymisji, zwolnić z funkcji,powołać rząd, rozwiązać parlament, wygrać wybory, 
        przegrać wybory, zaprzysiężenie, referendum, spotkać się dyplomatycznie, negocjacje międzynarodowe, zawrzeć porozumienie, zerwać rozmowy, sankcje, embargo,
        wezwać ambasadora, wydalić dyplomatę, ogłosić stan wyjątkowy, zamknąć granice, wprowadzić zakaz, protestować, manifestacja, demonstracja, strajk, strajk generalny,
        pikieta, marsz, blokada dróg, blokada parlamentu, presja społeczna, wystąpienie związków zawodowych, ultimatum wobec rządu, postulaty protestujących

    6. BRAK_ZDARZENIA (Wszystko inne)
    - opinie, komentarze, publicystyka
    - sondaże, badania opinii
    - zapowiedzi wydarzeń
    - sport (jeśli nie kryminalny)
    - pogoda
    - poradniki, ciekawostki
    - analiza, wywiad, felieton
    - stan rzeczy, trendy, prognozy

    Zwróć wynik w formacie JSON zawierającym listę obiektów z polami: 'text' i 'label'.

    Lista zdań do analizy:
    {json.dumps(sentences_list, ensure_ascii=False)}
    """

    try:
        result = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=BatchResponse
            )
        )
        parsed = json.loads(result.text)
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

    for i in range(0, total_items, BATCH_SIZE):
        batch_objects = raw_data[i : i + BATCH_SIZE]
        batch_texts = [obj.get("Zdanie", "") for obj in batch_objects]
        
        print(f"Przetwarzanie...")
        
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
    # run_classification()
    # run_balancing()
       

if __name__ == "__main__":
    main()