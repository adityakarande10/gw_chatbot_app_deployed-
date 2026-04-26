# 🎓 GW University RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built on GW University bulletin PDFs. Ask it anything about GW programs, tuition, admissions, academic policies, and more — it answers using real GW documents.

## 🌐 Live Demo

👉 **[https://applicationchatbot.streamlit.app/](https://applicationchatbot.streamlit.app/)**

No setup required. The knowledge base and API key are pre-configured.

---

## 📸 Preview

> The chatbot loads GW University bulletin PDFs as its knowledge base and answers questions with source attribution, powered by LLaMA 3.3 70B via the Groq API.

---

## 🏗️ Architecture

```
User Question
     ↓
Semantic Search (sentence-transformers: all-MiniLM-L6-v2)
     ↓
Retrieve Top-K Relevant Chunks from GW PDFs
     ↓
LLaMA 3.3 70B via Groq API generates a grounded answer
     ↓
Answer + Source attribution shown to user
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| PDF Parsing | PyPDF2 |
| Embedding Model | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Search | NumPy cosine similarity |
| LLM | LLaMA 3.3 70B via Groq API |
| Index Persistence | Python pickle |
| Deployment | Streamlit Community Cloud |

---

## 📚 Knowledge Base

The chatbot is pre-indexed on the following GW University bulletin PDFs:

- `academic-calendar.pdf`
- `arts-sciences.pdf`
- `business.pdf`
- `education-human-development.pdf`
- `engineering-applied-science.pdf`
- `fees-financial-regulations.pdf`
- `international-affairs.pdf`
- `medicine-health-sciences.pdf`
- `nursing.pdf`
- *(and more)*

---

## 💬 Sample Questions

- What is the minimum GPA required for graduate students at GW?
- How much is full-time undergraduate tuition?
- What happens if I get an Incomplete grade?
- How do I apply for a leave of absence?
- What engineering master programs does GW offer?
- What is the academic integrity policy at GW?
- How do I transfer credits to GW?

---

## 🚀 Run Locally

### Prerequisites

- Python 3.10, 3.11, or 3.12 (**do NOT use Python 3.14**)
- A free Groq API key → [https://console.groq.com](https://console.groq.com)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/adityakarande10/gw_chatbot_app_deployed-.git
cd gw_chatbot_app_deployed-

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run gw_chatbot_app_deployed.py
```

Opens at **http://localhost:8501**

Then:
1. Enter your Groq API key in the sidebar
2. Upload GW bulletin PDFs
3. Click **Process PDFs & Build Index**
4. Start chatting!

---

## ☁️ Deploy on Streamlit Cloud

1. Fork or push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New App**
3. Set the main file to `gw_chatbot_app_deployed.py`
4. Under **Advanced Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy**

---

## ⚙️ How It Works

1. **PDF Ingestion** — GW bulletin PDFs are parsed using PyPDF2 and split into overlapping text chunks (250 words, 50-word overlap).
2. **Embedding** — Each chunk is embedded using `all-MiniLM-L6-v2` from sentence-transformers.
3. **Index Persistence** — Embeddings and chunks are saved to `gw_index.pkl` so they load instantly on app restart.
4. **Retrieval** — At query time, cosine similarity finds the top-K most relevant chunks above a configurable similarity threshold.
5. **Generation** — The retrieved context is passed to LLaMA 3.3 70B via Groq, which generates a grounded, cited answer.

---

## 📁 Project Structure

```
gw_chatbot_app_deployed-/
├── gw_chatbot_app_deployed.py   # Main Streamlit app
├── gw_index.pkl                 # Pre-built vector index (auto-loaded)
├── requirements.txt             # Python dependencies
├── .gitignore
├── .devcontainer/               # GitHub Codespaces config
└── README.md
```

---

## 🔒 Security

- The Groq API key is stored in Streamlit Secrets and never exposed in the UI or committed to the repository.
- The pre-built index (`gw_index.pkl`) contains only text chunks and embeddings — no raw PDF data.

---

## 👤 Author

**Aditya Karande**
- GitHub: [@adityakarande10](https://github.com/adityakarande10)

---

## 📄 License

This project is for academic and educational purposes at The George Washington University.
