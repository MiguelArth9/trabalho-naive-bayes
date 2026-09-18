import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Classificação Naïve Bayes", layout="wide")

st.title("📊 Aplicação de Classificação Naïve Bayes")
st.markdown(
    "Interface interativa para testar os 3 modelos de Naïve Bayes treinados."
)

aba1, aba2, aba3 = st.tabs(
    ["GaussianNB", "CategoricalNB", "MultinomialNB (NLP)"]
)

# --- ABA 1: GAUSSIAN NB ---
with aba1:
    st.header("Modelo 1: Naïve Bayes Gaussiano")
    st.write("Insira 4 valores numéricos de entrada:")

    f1 = st.number_input("Atributo 1", value=0.0)
    f2 = st.number_input("Atributo 2", value=0.0)
    f3 = st.number_input("Atributo 3", value=0.0)
    f4 = st.number_input("Atributo 4", value=0.0)

    if st.button("Prever (Gaussiano)"):
        try:
            with open("modelo_gaussian.pkl", "rb") as f:
                m1 = pickle.load(f)
            pred = m1.predict([[f1, f2, f3, f4]])[0]
            proba = m1.predict_proba([[f1, f2, f3, f4]])[0]
            st.success(f"Classe Prevista: **{pred}**")
            st.write(
                f"Probabilidades — Classe 0: {proba[0]:.2f} | Classe 1: {proba[1]:.2f}"
            )
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")

# --- ABA 2: CATEGORICAL NB ---
with aba2:
    st.header("Modelo 2: Naïve Bayes Categórico")

    esc = st.selectbox(
        "Escolaridade", ["Medio", "Superior", "Pos-Graduacao"]
    )
    ren = st.selectbox("Faixa de Renda", ["Baixa", "Media", "Alta"])
    his = st.selectbox("Histórico de Crédito", ["Ruim", "Bom", "Excelente"])

    if st.button("Prever Risco (Categórico)"):
        try:
            with open("modelo_categorical.pkl", "rb") as f:
                m2 = pickle.load(f)
            df_input = pd.DataFrame(
                [[esc, ren, his]],
                columns=["escolaridade", "renda_faixa", "historico_credito"],
            )
            pred = m2.predict(df_input)[0]
            proba = m2.predict_proba(df_input)[0]
            rotulo = "Alto Risco (1)" if pred == 1 else "Baixo Risco (0)"
            st.success(f"Resultado: **{rotulo}**")
            st.write(
                f"Probabilidades — Baixo Risco: {proba[0]:.2f} | Alto Risco: {proba[1]:.2f}"
            )
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")

# --- ABA 3: MULTINOMIAL NB (NLP) ---
with aba3:
    st.header("Modelo 3: Naïve Bayes Multinomial (Processamento de Texto)")

    texto = st.text_area(
        "Digite uma avaliação ou texto:",
        "Excelente produto, entrega super rápida!",
    )

    if st.button("Analisar Sentimento"):
        try:
            with open("modelo_nlp.pkl", "rb") as f:
                m3 = pickle.load(f)
            pred = m3.predict([texto])[0]
            proba = m3.predict_proba([texto])[0]
            rotulo = "Positivo (1)" if pred == 1 else "Negativo (0)"
            st.success(f"Sentimento Previsto: **{rotulo}**")
            st.write(
                f"Probabilidades — Negativo: {proba[0]:.2f} | Positivo: {proba[1]:.2f}"
            )
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")