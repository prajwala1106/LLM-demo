\# 🧬 Gene Expression Q\&A Assistant (LLM-powered)



A Streamlit app that allows researchers to upload gene expression data (RNA-seq, DEGs, etc.) and query it using natural language with the help of an LLM.



\## ✨ Features

\- Upload gene expression datasets in CSV format.

\- Ask natural language questions like:

&nbsp; - "Which genes are most upregulated?"

&nbsp; - "Summarize enriched pathways in this dataset."

&nbsp; - "What biological processes are likely impacted?"

\- Get \*\*LLM-generated summaries and insights\*\*.

\- Simple visualization (logFC vs p-value scatter).



\## 🛠️ Tech Stack

\- \*\*Python, Streamlit, Pandas\*\*

\- \*\*OpenAI GPT (LLM integration)\*\*

\- \*\*Matplotlib\*\* for visualization



\## 🚀 Run Locally

```bash

pip install -r requirements.txt

streamlit run app.py



