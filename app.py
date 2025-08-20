import numpy as np
import streamlit as st
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Gene Expression Q&A Assistant", page_icon="🧬", layout="wide")

st.title("🧬 Gene Expression Q&A Assistant (LLM-powered)")
st.write("Upload RNA-Seq or gene expression data and ask natural language questions.")

# Example CSV for users
example_csv = """gene,logFC,pvalue
K21925,2.1,0.0005
K20618,1.7,0.0012
K18108,-1.5,0.0021
K07385,2.8,0.0001
K12467,-2.2,0.0034
K21374,1.9,0.0009
"""

st.download_button(
    label="📥 Download Example CSV",
    data=example_csv,
    file_name="example_gene_expression.csv",
    mime="text/csv"
)

# Upload CSV
uploaded_file = st.file_uploader("Upload your expression CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    st.write("### Data Preview")
    st.dataframe(df.head())

    # Quick plot if required columns exist
    if "logfc" in df.columns and "pvalue" in df.columns:
        st.write("### Volcano-like Plot (logFC vs p-value)")
        fig, ax = plt.subplots()
        ax.scatter(df["logfc"], -np.log10(df["pvalue"]), alpha=0.5)
        ax.set_xlabel("logFC")
        ax.set_ylabel("-log10(pvalue)")
        st.pyplot(fig)
    else:
        st.warning("CSV must have 'logFC' (or similar) and 'pvalue' columns.")

    # Input for user question
    question = st.text_area("Ask a question about this dataset:")

    if question:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        # Convert dataframe to text summary (first 20 rows only)
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
