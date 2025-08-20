import streamlit as st
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gene Expression Q&A Assistant", page_icon="🧬", layout="wide")

st.title("🧬 Gene Expression Q&A Assistant (LLM-powered)")
st.write("Upload RNA-Seq or gene expression data and ask natural language questions.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your expression CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Quick plot
    if "logFC" in df.columns:
        st.write("### Volcano-like Plot (logFC vs p-value)")
        fig, ax = plt.subplots()
        ax.scatter(df["logFC"], -np.log10(df["pvalue"]), alpha=0.5)
        ax.set_xlabel("logFC")
        ax.set_ylabel("-log10(pvalue)")
        st.pyplot(fig)

    # Input for user question
    question = st.text_area("Ask a question about this dataset:")

    if question:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        # Convert dataframe to text summary (small slice to avoid token explosion)
        summary = df.head(20).to_string()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a genomics expert interpreting gene expression data."},
                {"role": "user", "content": f"Dataset preview:\n{summary}\n\nQuestion: {question}"}
            ]
        )

        st.write("### 🔍 LLM Answer:")
        st.write(response.choices[0].message.content)
