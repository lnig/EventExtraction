import streamlit as st
import spacy
import pandas as pd
import plotly.express as px

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="NLP Event Analyzer",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# --- CSS (Wygląd) ---
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- ŁADOWANIE MODELI ---
@st.cache_resource
def load_classifier_model():
    try:
        return spacy.load("models/output_herbert/model-best") 
    except OSError:
        return None

@st.cache_resource
def load_grammar_model():
    try:
        if not spacy.util.is_package("pl_core_news_lg"):
            spacy.cli.download("pl_core_news_lg")
        return spacy.load("pl_core_news_lg")
    except Exception as e:
        st.error(f"Błąd ładowania modelu gramatycznego: {e}")
        return None

# --- FUNKCJA EKSTRAKCJI ---
def extract_details(doc):
    data = {"TRIGGER": "-", "KTO": "-", "CO": "-", "GDZIE": "-", "KIEDY": "-"}
    try:
        root = [token for token in doc if token.dep_ == "ROOT"][0]
        data["TRIGGER"] = root.lemma_
    except IndexError:
        return data

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
            elif any(l in ["date", "time"] for l in ents_labels) or any(w in subtree_text for w in ["wczoraj", "dziś", "jutro", "roku"]):
                data["KIEDY"] = subtree_text
            elif " w " in " "+subtree_text or " na " in " "+subtree_text: 
                if data["GDZIE"] == "-": data["GDZIE"] = subtree_text

    for ent in doc.ents:
        if ent.label_ in ["placeName", "geogName", "GPE", "LOC"] and data["GDZIE"] == "-":
            data["GDZIE"] = ent.text
        if ent.label_ in ["date", "time"] and data["KIEDY"] == "-":
            data["KIEDY"] = ent.text
    return data

# --- INICJALIZACJA ---
nlp_cat = load_classifier_model()
nlp_gram = load_grammar_model()

# --- GŁÓWNY INTERFEJS ---
st.title("🕵️‍♂️ NLP News Intelligence")

default_text = """Złodziej ukradł portfel pasażerowi w tramwaju. Policja szybko ujęła sprawcę. 
Premier odwołał ministra zdrowia wczoraj wieczorem. 
Orlen ogłosił rekordowe zyski, a akcje poszybowały w górę.
Wypadek autokaru pod Krakowem zablokował autostradę A4.
Huragan zerwał dachy z domów w województwie pomorskim.
To był zwykły, słoneczny dzień bez żadnych wydarzeń."""

text_input = st.text_area("Wklej treść artykułu:", default_text, height=250)

# --- PANEL STEROWANIA ---
col_opt, col_btn = st.columns([3, 1])
with col_opt:
    st.write("") 
    hide_none = st.checkbox("Ukryj zdania bez wykrytych zdarzeń (BRAK_ZDARZENIA) na liście wyników", value=True)
with col_btn:
    run_button = st.button("🚀 Analizuj", type="primary", use_container_width=True)

# --- ANALIZA ---
if run_button and text_input:
    if not nlp_cat or not nlp_gram:
        st.error("Błąd: Modele nie są dostępne.")
        st.stop()

    with st.spinner("Przetwarzanie..."):
        doc_whole = nlp_gram(text_input)
        sentences = list(doc_whole.sents)
        
        results = []
        stats = {"BRAK_ZDARZENIA": 0, "PRZESTEPSTWO": 0, "POLITYKA": 0, "BIZNES": 0, "KATASTROFA": 0, "WYPADEK": 0}
        
        for sent in sentences:
            sent_text = sent.text.strip()
            if len(sent_text) < 5: continue
            
            # Klasyfikacja
            doc_cat = nlp_cat(sent_text)
            scores = doc_cat.cats
            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]
            
            if best_label in stats:
                stats[best_label] += 1
            
            # Ekstrakcja
            doc_sent_gram = sent.as_doc() 
            details = extract_details(doc_sent_gram)
            
            results.append({
                "text": sent_text,
                "label": best_label,
                "score": best_score,
                "scores_full": scores,
                "details": details
            })

    # --- WYNIKI: STATYSTYKI ---
    st.divider()
    st.subheader("📊 Statystyki")
    cols = st.columns(6)
    cols[0].metric("Wszystkie", len(sentences))
    cols[1].metric("Przestępstwa", stats.get("PRZESTEPSTWO", 0))
    cols[2].metric("Polityka", stats.get("POLITYKA", 0))
    cols[3].metric("Biznes", stats.get("BIZNES", 0))
    cols[4].metric("Wypadki", stats.get("WYPADEK", 0))
    cols[5].metric("Katastrofy", stats.get("KATASTROFA", 0))
    
    # --- WYKRES KOŁOWY ---
    df_stats = pd.DataFrame(list(stats.items()), columns=["Kategoria", "Liczba"])
    if not df_stats.empty and df_stats['Liczba'].sum() > 0:
        fig = px.pie(
            df_stats, values='Liczba', names='Kategoria', hole=0.4, height=450
        )
        fig.update_layout(legend=dict(font=dict(size=18), orientation="v"))
        fig.update_traces(textinfo='value+percent', textposition='inside', textfont_size=16)
        st.plotly_chart(fig, use_container_width=True)
    else:
         st.info("Brak danych do wyświetlenia na wykresie.")

    # --- WYNIKI: LISTA SZCZEGÓŁOWA (ULEPSZONA) ---
    st.subheader("📝 Szczegółowa Lista Wyników")
    
    found_important = False
    
    for item in results:
        label = item['label']
        if hide_none and label == "BRAK_ZDARZENIA":
            continue
            
        found_important = True
        score = item['score']
        text = item['text']
        
        icon_map = {
            "PRZESTEPSTWO": "🚓", "POLITYKA": "🏛️", "BIZNES": "💼", 
            "WYPADEK": "🚑", "KATASTROFA": "🔥", "BRAK_ZDARZENIA": "⚪"
        }
        icon = icon_map.get(label, "❓")
        
        with st.expander(f"{icon} **{label}** ({score:.1%}): {text}", expanded=(label != "BRAK_ZDARZENIA")):
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.markdown("**Rozkład prawdopodobieństwa:**")
                
                # PRZYGOTOWANIE DANYCH DO NOWEGO WYKRESU
                df_scores = pd.DataFrame(list(item['scores_full'].items()), columns=["Kategoria", "Pewnosc"])
                df_scores = df_scores.sort_values(by="Pewnosc", ascending=True)
                
                # WYKRES POZIOMY PLOTLY
                fig_bar = px.bar(
                    df_scores, 
                    x="Pewnosc", 
                    y="Kategoria", 
                    orientation='h',
                    text_auto='.1%',
                    range_x=[0, 1]
                )
                
                # Stylizacja wykresu słupkowego
                fig_bar.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title=None,
                    yaxis_title=None,
                    showlegend=False
                )
                fig_bar.update_traces(textfont_size=12, marker_color='#4CAF50')
                st.plotly_chart(fig_bar, use_container_width=True)

            with c2:
                st.markdown("**Szczegóły zdarzenia:**")
                if any(v != "-" for v in item['details'].values()):
                    df_det = pd.DataFrame(item['details'].items(), columns=["Pole", "Wartość"])
                    st.data_editor(
                        df_det, 
                        hide_index=True, 
                        use_container_width=True, 
                        disabled=True
                    )
                else:
                    st.info("Nie udało się wyodrębnić szczegółów (Kto/Co/Gdzie).")

    if not found_important and hide_none:
        st.info("Wszystkie zdania zostały zaklasyfikowane jako BRAK_ZDARZENIA. Odznacz checkbox powyżej i kliknij ponownie 'Analizuj', aby je zobaczyć.")

    # --- EKSPORT ---
    if results:
        st.divider()
        export_data = []
        for r in results:
            row = {"Zdanie": r['text'], "Etykieta": r['label'], "Pewnosc": r['score']}
            row.update(r['details'])
            export_data.append(row)
            
        st.download_button(
            label="📥 Pobierz wszystkie wyniki (CSV)",
            data=pd.DataFrame(export_data).to_csv(index=False).encode('utf-8'),
            file_name='analiza.csv',
            mime='text/csv',
        )