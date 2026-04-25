# 🎓 GW University RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built on GW University bulletin PDFs.

## 🌐 Live Demo
👉 **[Click here to open the chatbot](#)** ← replace with your Streamlit URL after deploying

---

## Architecture

```
User Question
     ↓
Semantic Search (sentence-transformers: all-MiniLM-L6-v2)
     ↓
Retrieve Top-K Relevant Chunks from GW PDFs
     ↓
Groq LLaMA 3.3 70B generates a grounded answer
     ↓
Answer + Source attribution shown to user
```

---

## Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| PDF Parsing | PyPDF2 |
| Embedding Model | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | NumPy cosine similarity |
| LLM | LLaMA 3.3 70B via Groq API |
| Index Persistence | Python pickle |

---

## 🚀 Option A — Use the Live Deployed App (Recommended)

Just open the link above. No setup needed. The knowledge base and API key are pre-configured.

---

## 🖥️ Option B — Run Locally

### Prerequisites
- Python 3.10, 3.11, or 3.12 (**do NOT use Python 3.14**)
- A free Groq API key → https://console.groq.com

### Steps

```bash
# 1. Navigate into the project folder
cd gw-chatbot

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run gw_chatbot_app.py
```

Opens at **http://localhost:8501**

Then:
1. Enter your Groq API key in the sidebar
2. Upload PDFs from the `Dataset/` folder
3. Click **Process PDFs & Build Index**
4. Start chatting!

---

## Sample Questions

- What is the minimum GPA required for graduate students at GW?
- How much is full-time undergraduate tuition?
- What happens if I get an Incomplete grade?
- How do I apply for a leave of absence?
- What engineering master programs does GW offer?
- What is the academic integrity policy at GW?
