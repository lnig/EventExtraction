import streamlit as st
import spacy
import pandas as pd

st.set_page_config(page_title="Ekstrakcja Zdarzeń NLP", layout="wide")

@st.cache_resource
def load_classifier_model():
    try:
        return spacy.load("models/output_herbert/model-best") 
    except OSError:
        st.error("Błąd: Nie znaleziono modelu w 'models/output_herbert/model-best'.")
        return None

@st.cache_resource
def load_grammar_model():
    try:
        return spacy.load("pl_core_news_lg")
    except OSError:
        st.warning("Pobieram model pl_core_news_lg...")
        spacy.cli.download("pl_core_news_lg")
        return spacy.load("pl_core_news_lg")

nlp_cat = load_classifier_model()
nlp_gram = load_grammar_model()

def extract_details(text):
    if not nlp_gram:
        return {}, None

    doc = nlp_gram(text)
    data = {"TRIGGER": "-", "KTO": "-", "CO": "-", "GDZIE": "-", "KIEDY": "-"}

    try:
        root = [token for token in doc if token.dep_ == "ROOT"][0]
        data["TRIGGER"] = root.lemma_
    except IndexError:
        return data, doc

    for child in root.children:
        subtree_text = " ".join([t.text for t in child.subtree])
        
        if child.dep_ == "nsubj":
            data["KTO"] = subtree_text
        elif child.dep_ in ["obj", "nsubj:pass"]:
            data["CO"] = subtree_text
        elif child.dep_ == "obl":
            ents_labels = [e.ent_type_ for e in child.subtree if e.ent_type_]
            
            if any(l in ["placeName", "geogName", "GPE", "LOC"] for l in ents_labels):
                data["GDZIE"] = subtree_text
            elif any(l in ["date", "time"] for l in ents_labels) or any(w in subtree_text for w in ["wczoraj", "dziś", "roku"]):
                data["KIEDY"] = subtree_text
            elif " w " in " "+subtree_text or " na " in " "+subtree_text: 
                if data["GDZIE"] == "-": data["GDZIE"] = subtree_text

    for ent in doc.ents:
        if ent.label_ in ["placeName", "geogName", "GPE", "LOC"] and data["GDZIE"] == "-":
            data["GDZIE"] = ent.text
        if ent.label_ in ["date", "time"] and data["KIEDY"] == "-":
            data["KIEDY"] = ent.text

    return data, doc

st.title("🕵️‍♂️ System Analizy Newsów (Full Text)")
st.markdown("""
Model hybrydowy: **Spacy TextCat (HerBERT)** + **Universal Dependencies**.
Wklej cały artykuł - system automatycznie podzieli go na zdania i przeanalizuje każde z osobna.
""")

default_text = """Złodziej ukradł portfel pasażerowi w tramwaju. Policja szybko ujęła sprawcę. 
Premier odwołał ministra zdrowia wczoraj wieczorem. To była trudna decyzja. 
Eksperci przewidują wzrost inflacji."""

text_input = st.text_area("Wklej treść artykułu:", default_text, height=150)

filter_events = st.checkbox("Pokaż tylko wykryte zdarzenia (ukryj BRAK_ZDARZENIA)", value=False)

if st.button("Analizuj Tekst") and nlp_cat and nlp_gram:
    
    st.write("---")
    
    doc_whole = nlp_gram(text_input)
    sentences = list(doc_whole.sents)
    
    st.info(f"Wykryto zdań: {len(sentences)}")
    
    found_any = False

    for i, sent in enumerate(sentences):
        sent_text = sent.text.strip()
        if not sent_text: continue
        
        doc_cat = nlp_cat(sent_text)
        scores = doc_cat.cats
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        
        if filter_events and best_label == "BRAK_ZDARZENIA":
            continue
            
        found_any = True
        
        icon = "🚨" if best_label in ["PRZESTEPSTWO", "KATASTROFA", "WYPADEK"] else "📢"
        if best_label == "BRAK_ZDARZENIA": icon = "⚪"
        
        expander_title = f"{icon} [{best_label}] {sent_text[:60]}..."
        
        with st.expander(expander_title, expanded=(best_label != "BRAK_ZDARZENIA")):
            st.markdown(f"**Pełne zdanie:** {sent_text}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.caption("Klasyfikacja")
                color = "green" if best_score > 0.8 else "orange"
                st.markdown(f"Kategoria: :{color}[**{best_label}**]")
                st.progress(best_score)
                if best_label != "BRAK_ZDARZENIA":
                    st.bar_chart(scores, height=150)

            with col2:
                st.caption("Ekstrakcja szczegółów")
                details, doc_gram_sent = extract_details(sent_text)
                
                if best_label != "BRAK_ZDARZENIA" or any(v != "-" for v in details.values()):
                    df_details = pd.DataFrame(details.items(), columns=["Slot", "Wartość"])
                    st.dataframe(df_details, hide_index=True, use_container_width=True)
                else:
                    st.info("Brak szczegółów do wyekstrahowania.")

            if best_label != "BRAK_ZDARZENIA":
                st.caption("Struktura gramatyczna:")
                html = spacy.displacy.render(doc_gram_sent, style="dep", options={"compact": True, "distance": 90, "bg": "#f0f2f6"})
                st.components.v1.html(html, height=200, scrolling=True)

    if not found_any and filter_events:
        st.warning("Nie znaleziono żadnych istotnych zdarzeń w tekście (wszystkie zaklasyfikowano jako BRAK_ZDARZENIA). Odznacz filtr, aby zobaczyć wszystko.")