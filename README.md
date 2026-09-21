<div align="center">

```
███╗   ███╗██╗███╗   ██╗██████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
████╗ ████║██║████╗  ██║██╔══██╗██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
██╔████╔██║██║██╔██╗ ██║██║  ██║██║  ███╗██║   ██║███████║██████╔╝██║  ██║
██║╚██╔╝██║██║██║╚██╗██║██║  ██║██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
██║ ╚═╝ ██║██║██║ ╚████║██████╔╝╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
```

### *When the mind needs a guardian, science answers the call.*

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-XLM--RoBERTa-FFD21E?style=for-the-badge)](https://huggingface.co)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge)](https://groq.com)
[![LangChain](https://img.shields.io/badge/🦜🔗_LangChain-RAG_Orchestration-1C3C3C?style=for-the-badge)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-6C3EF4?style=for-the-badge)](https://www.trychroma.com/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-008080?style=for-the-badge)](https://shap.readthedocs.io)

</div>

---

> ⚠️ **Before anything else:** MindGuard is a research and portfolio engineering project — **not** a licensed medical device, and **not** a substitute for professional mental health care. If you or someone you know is in crisis, please reach out to a real, trained human: see the [full disclaimer and crisis helplines](#-ethical-disclaimer) below.

---

<br>

> **MindGuard** doesn't just chat back — it *classifies*, *retrieves*, *explains*, and *remembers*. Every message is scored by a fine-tuned XLM-RoBERTa classifier, matched against a curated, source-cited library of coping strategies via a retrieval-augmented pipeline, mathematically explained with SHAP, and answered by a clinically-prompted Groq LLM that recalls the last few turns of conversation.

<br>

## ◈ What Makes This Different

Most mental-health chatbot demos are a thin prompt wrapped around an LLM. MindGuard is a full pipeline of dedicated, purpose-built components:

| Capability | Generic Chatbot | **MindGuard** |
|---|---|---|
| Response generation | Raw LLM API call | Groq Llama-3.3-70B with a strict clinical system prompt |
| Emotion detection | Guessed from LLM output | Dedicated fine-tuned **XLM-RoBERTa** classifier — 35 distinct labels |
| Grounded coping strategies | None, or hallucinated | **RAG** over a 133-entry, source-cited CBT knowledge base, orchestrated with **LangChain** + ChromaDB |
| Why did it predict that? | Black box | **SHAP** word-level attribution rendered as an interactive HTML report |
| Risk triage | None | Predicted emotion mapped to a High / Medium / Low risk badge |
| Conversation memory | Varies | Last 3 turns pulled from SQLite and woven into every new prompt |
| Voice input | None | Speech transcribed via Groq's hosted **Whisper-large-v3** API |
| Clinical audit trail | None | Full SQLite session history — every turn, emotion, and risk level, timestamped |
| XAI dashboard | None | Dedicated clinician-facing tab rendering the latest SHAP report |

<br>

---

## 📌 Table of Contents

- [System Architecture](#-system-architecture)
- [The Request Pipeline, Step by Step](#-the-request-pipeline-step-by-step)
- [Emotion Model & Risk Classification](#-emotion-model--risk-classification)
- [Explainability (SHAP)](#-explainability-shap)
- [RAG Knowledge Base](#-rag-knowledge-base)
- [Voice Input](#-voice-input)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Training Your Own Model](#-training-your-own-model)
- [Running the App](#-running-the-app)
- [Roadmap](#-roadmap)
- [Engineering Notes](#-engineering-notes)
- [Ethical Disclaimer](#-ethical-disclaimer)
- [License](#-license)
- [Author](#-author)

---

## ◈ System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER (Streamlit)                   │
│  💬 Chat Companion (badges + inline SHAP)   📊 Clinical Dashboard    │
└──────────────────────────────┬──────────────────────────────────────┬─┘
                                │ text                          voice │
                                │                                     ▼
                                │                       ┌───────────────────────┐
                                │                       │ Groq Whisper-large-v3 │
                                │                       │  (cloud transcription)│
                                │                       └───────────┬───────────┘
                                └──────────────────┬─────────────────┘
                                                   ▼
                                     ┌─────────────────────────────┐
                                     │  XLM-RoBERTa Classifier     │
                                     │  35 emotion labels → risk   │
                                     └───────────────┬─────────────┘
                                       ┌──────────────┼──────────────┐
                                       ▼                             ▼
                          ┌─────────────────────┐      ┌───────────────────────────┐
                          │   SHAP Explainer    │      │  LangChain RAG Retriever  │
                          │  (word attribution) │      │ (BAAI/bge-base-en-v1.5)   │
                          └─────────────────────┘      └─────────────┬─────────────┘
                                                                     ▼
                                                        ┌───────────────────────────┐
                                                        │ SQLite — last 3 turns     │
                                                        │ (conversation memory)     │
                                                        └─────────────┬─────────────┘
                                                                      ▼
                                                        ┌───────────────────────────┐
                                                        │  Groq Llama-3.3-70B       │
                                                        │  (clinical response gen.) │
                                                        └─────────────┬─────────────┘
                                                                      ▼
                                                        ┌───────────────────────────┐
                                                        │ SQLite: save interaction  │
                                                        │ artifacts/shap_report.html│
                                                        └───────────────────────────┘
```

---

## 🔄 The Request Pipeline, Step by Step

Every message — typed or spoken — runs through the exact same six-step pipeline inside `MindGuardChatbot.generate_response()`:

1. **The Psychologist** — the XLM-RoBERTa classifier predicts one of 35 emotion labels and derives a risk level
2. **The Librarian** — a LangChain `VectorStoreRetriever` embeds the message and pulls the single most relevant coping strategy, filtered to the predicted emotion where possible
3. **The Memory Bank** — the last 3 turns of that session are pulled from SQLite for conversational continuity
4. **Prompt Assembly** — the diagnosis, retrieved strategy, and recent history are woven into one augmented prompt
5. **The Mouth** — a LangChain LCEL chain (`ChatPromptTemplate | ChatGroq | StrOutputParser`) drives Groq's Llama-3.3-70B to generate the final empathetic response under a strict clinical system prompt
6. **The Record** — the full turn (message, response, emotion, risk) is saved back to SQLite for the audit trail and dashboard

---

## 🧠 Emotion Model & Risk Classification

### 35-Class Emotion Taxonomy

The classifier is a fine-tuned `xlm-roberta-base` trained on a merge of three sources — a clinical mental-health text dataset, Google's **GoEmotions**, and a third HuggingFace parquet export — combined into one 35-label taxonomy:

**Clinical labels (7)** — `Suicidal`, `Depression`, `Anxiety`, `Bipolar`, `Stress`, `Personality disorder`, `Normal`

**GoEmotions fine-grained labels (28)** — admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, neutral, optimism, pride, realization, relief, remorse, sadness, surprise

> **On "multilingual":** the model is built on XLM-RoBERTa's multilingual pretrained backbone (100+ languages), but the fine-tuning data assembled by `cleaner.py` is English-language text. The model *inherits* multilingual representations from pretraining, but multilingual clinical accuracy hasn't been separately evaluated in this repo.

### Risk Mapping (as actually implemented)

`determine_risk_level()` does a simple lowercase keyword lookup on the predicted label — it is **not** a learned or graded score:

| Predicted label | Risk level |
|---|---|
| `Suicidal`, `Depression`, `Personality disorder`, `grief` | **High** |
| `Stress`, `Anxiety`, `anger`, `fear`, `nervousness` | **Medium** |
| Everything else — including `Bipolar`, `sadness`, `Normal`, and the remaining GoEmotions | **Low** |

Two keyword slots (`panic`, `severe anxiety` → High; `burnout` → Medium) are already wired into the logic but currently unreachable, since those exact strings aren't in the 35-class label set — they're reserved for a future label expansion.

---

## 🔬 Explainability (SHAP)

MindGuard wraps SHAP's `Explainer` around a Hugging Face `text-classification` pipeline running the same XLM-RoBERTa weights (with `top_k=None` so every one of the 35 class scores is available), then:

1. Runs the full Game-Theoretic SHAP computation on the input text
2. Automatically selects the single emotion class the model was most confident in
3. Renders `shap.plots.text` — an inline, color-highlighted HTML view where **red** words pushed the prediction toward that emotion and **blue** words pushed against it
4. Saves the result to `artifacts/shap_report.html`, which is then embedded both inline under the chat message and in a dedicated **XAI Report** tab on the Clinical Dashboard

```
Input:  "I have a massive presentation tomorrow and my chest is tight."
                │                    │                        │
                ▼                    ▼                        ▼
           [neutral]           [HIGH IMPACT]            [HIGH IMPACT]
                               "presentation"           "chest is tight"
                                    │                        │
                                    └──────────┬─────────────┘
                                               ▼
                                    Predicted: ANXIETY
                                    Risk Level: MEDIUM ⚠️
```

---

## 📚 RAG Knowledge Base

`data/knowledge_base/coping_strategies.json` ships **133 curated, source-cited coping strategies**, each with a primary emotion tag, target risk level, category, tags, and a clinical source reference (e.g. *"Standard Cognitive Behavioral Therapy (CBT)"*, *"Clinical Breathwork Guidelines"*). Categories span far beyond generic advice — Imposter Syndrome, Workplace Burnout, Financial Stress, Caregiver Burnout, Digital Overload, Existential Dread, and more.

- The whole ingestion → retrieval pipeline is built on **LangChain**: each strategy is wrapped as a LangChain `Document`, embedded with `HuggingFaceEmbeddings` (`BAAI/bge-base-en-v1.5`), and stored in a `langchain_chroma.Chroma` vector store — a LangChain-native wrapper around the same persistent **ChromaDB** collection (`clinical_guidelines`), with rich metadata for filtering
- `build_vector_db.py` reads `coping_strategies.json`, builds the `Document` list, and upserts it into the vector store (safe to re-run — no duplicates)
- `retriever.py` wraps the same vector store as a LangChain `VectorStoreRetriever`, optionally filtered strictly by the emotion the classifier just predicted (`search_kwargs={"filter": {"emotion": ...}}`), and returns the single best-matching strategy via `.invoke()`
- If nothing matches (empty DB, or an overly strict filter), it falls back to a safe, conversational prompt rather than failing silently

---

## 🎙️ Voice Input

The chat sidebar exposes Streamlit's native `st.audio_input()` recorder. Recorded audio is:

1. Written to a temporary `.wav` file, de-duplicated by byte-size fingerprint (to stop Streamlit's automatic rerun from reprocessing the same recording twice)
2. Sent to **Groq's hosted `whisper-large-v3` model** over the Groq API (with a domain-specific transcription prompt and `temperature=0.0` to reduce hallucinated words) — this is a **cloud API call**, not a locally-running Whisper model
3. Fed into the exact same `generate_response()` pipeline as typed text

---

## 🛠️ Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| **LLM** | Groq — Llama-3.3-70B-Versatile | Clinical system prompt, low-latency inference |
| **Emotion Model** | XLM-RoBERTa (fine-tuned) | 35-class classifier, class-weighted loss, macro-F1 selection |
| **Speech-to-Text** | Groq — Whisper-large-v3 | Cloud API, not offline |
| **Explainability** | SHAP `Explainer` | Word-level, game-theoretic attribution |
| **RAG Orchestration** | **LangChain** (`langchain-chroma`, `langchain-huggingface`, `langchain-groq`) | Documents, vector store, retriever, and LCEL generation chain |
| **Vector DB / Embeddings** | ChromaDB + `BAAI/bge-base-en-v1.5` | 133-entry clinical strategy library |
| **UI Framework** | Streamlit | Chat companion + clinical dashboard |
| **Database** | SQLite | Session history, conversation memory, dashboard analytics |
| **ML Utilities** | PyTorch, Transformers, `datasets`, `accelerate` | Training & inference stack |

---

## 📂 Project Structure

```
MindGuard-AI-Mental-Health-System/
├── app/
│   ├── main.py                       # Entry point — sidebar navigation/routing
│   ├── api.py                        # @st.cache_resource loaders for the bot & SHAP engine
│   └── components/
│       ├── chat_ui.py                # Chat interface — badges, SHAP embed, voice input
│       └── dashboard_ui.py           # Analytics tab + XAI Report tab
│
├── src/
│   ├── chatbot/
│   │   └── groq_bot.py               # MindGuardChatbot — the 6-step orchestration pipeline
│   ├── core_model/
│   │   ├── predict.py                # Loads XLM-R weights, predicts emotion + risk
│   │   └── train.py                  # Fine-tunes xlm-roberta-base with class-weighted loss
│   ├── explainability/
│   │   └── shap_explainer.py         # SHAP Explainer → HTML report generation
│   ├── rag_engine/
│   │   ├── build_vector_db.py        # LangChain: JSON → Documents → HuggingFaceEmbeddings → Chroma
│   │   └── retriever.py              # LangChain VectorStoreRetriever + emotion-filtered retrieval
│   ├── audio/
│   │   └── speech_to_text.py         # Groq Whisper-large-v3 transcription wrapper
│   ├── database/
│   │   └── db_operations.py          # SQLite schema, save/read chat history
│   └── preprocessing/
│       └── cleaner.py                # Merges 3 raw datasets into one training CSV
│
├── data/
│   └── knowledge_base/
│       └── coping_strategies.json    # 133 source-cited CBT/clinical coping strategies
│
├── notebooks/
│   └── helper.ipynb                  # XLM-RoBERTa fine-tuning notebook (Colab/Kaggle GPU)
│
├── requirements.txt
├── .gitignore
└── README.md
```

> `artifacts/` (model weights, ChromaDB store, SQLite file) and `data/raw/` (temp voice recordings, raw training CSVs) are git-ignored and generated locally — see [Getting Started](#-getting-started).

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.10
```

MindGuard needs three things before its first launch that are **not** included in the repository (all git-ignored by design):

1. A `GROQ_API_KEY` (free tier at [console.groq.com](https://console.groq.com))
2. Fine-tuned XLM-RoBERTa weights at `artifacts/xlmr_weights/final_mindguard_model/`
3. A populated ChromaDB store at `artifacts/chroma_db/`

### 1 · Clone & install

```bash
git clone https://github.com/MohitParmar78/MindGuard-AI-Mental-Health-System.git
cd MindGuard-AI-Mental-Health-System
pip install -r requirements.txt
```

### 2 · Configure secrets

Create a `.env` file at the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3 · Provide the model weights

Train your own (see [Training Your Own Model](#-training-your-own-model) below) or place pretrained weights so the following files exist:

```
artifacts/xlmr_weights/final_mindguard_model/
├── config.json
├── pytorch_model.bin  (or model.safetensors)
├── tokenizer_config.json
└── vocab.json
```

### 4 · Build the RAG vector store

```bash
python -m src.rag_engine.build_vector_db
```

This reads `data/knowledge_base/coping_strategies.json`, wraps each entry as a LangChain `Document`, downloads `BAAI/bge-base-en-v1.5` on first run, and populates the LangChain-managed `Chroma` store at `artifacts/chroma_db/`. The SQLite database at `artifacts/database/` is created automatically the first time the app runs — no manual step needed.

### 5 · Launch

```bash
streamlit run app/main.py
```

Navigate to `http://localhost:8501`.

---

## 🔑 Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | **Yes** | Powers both the Llama-3.3-70B chat completions and the Whisper-large-v3 audio transcription — every LLM/voice feature depends on it |

This is the only environment variable read anywhere in the codebase.

---

## 🎓 Training Your Own Model

`notebooks/helper.ipynb` is built to run on a Colab/Kaggle GPU runtime:

1. `src/preprocessing/cleaner.py` merges the clinical dataset, GoEmotions, and a HuggingFace parquet export into `data/processed/master_training_data.csv`, dropping corrupted/numeric labels
2. `src/core_model/train.py` fine-tunes `xlm-roberta-base` for 5 epochs (`lr=3e-5`, batch size 16, 500 warmup steps) using a **custom `ImbalancedTrainer`** that applies `sklearn`-computed class weights to the cross-entropy loss
3. Model selection uses **macro-F1**, not accuracy — this specifically forces the model to learn rare clinical classes rather than just predicting the majority label
4. The notebook zips the final weights so they can be downloaded and dropped into `artifacts/xlmr_weights/final_mindguard_model/` locally

```bash
python -m src.core_model.train
```

---

## ▶️ Running the App

```bash
streamlit run app/main.py
```

The sidebar switches between the **💬 Chat Companion** (text or voice, live badges, inline SHAP) and the **📊 Clinical Dashboard** (analytics + XAI report viewer). Both read from and write to the same local SQLite database.

---

## 🗺️ Roadmap

**Shipped**
- [x] XLM-RoBERTa 35-emotion fine-tuned classifier with class-weighted, macro-F1-selected training
- [x] SHAP word-level explainability with an interactive HTML report
- [x] Groq LLM integration with a clinical system prompt
- [x] **RAG retrieval** over a 133-entry, source-cited CBT knowledge base (ChromaDB)
- [x] **Short-term conversation memory** (last 3 turns recalled every response)
- [x] Voice input via Groq-hosted Whisper-large-v3
- [x] SQLite audit trail + clinical analytics dashboard
- [x] Keyword-based High/Medium/Low risk badge system

**Potential next steps**
- [ ] Expand the risk-keyword mapping to explicitly cover every clinical/GoEmotions label (e.g. `Bipolar`, `sadness` currently default to Low)
- [ ] Multi-user auth with persistent per-user profiles (sessions are currently a single hardcoded demo ID)
- [ ] A FastAPI service layer to decouple inference from the Streamlit UI (`fastapi`/`uvicorn` are already in `requirements.txt`, not yet wired up)
- [ ] Longitudinal mood tracking with trend analysis
- [ ] Formal offline evaluation of classifier accuracy/F1 on a held-out test set, published in this README

---

## 🔍 Engineering Notes

- **Risk levels are keyword-driven, not learned.** `Bipolar` and `sadness` — both real classifier outputs — currently resolve to **Low** risk because they aren't in the `high_risk`/`medium_risk` keyword lists. Worth knowing before treating the risk badge as a complete clinical signal.
- `fastapi`, `uvicorn`, and `plotly` are listed in `requirements.txt` but aren't imported anywhere in the current codebase — they appear to be reserved for a planned API layer / richer charting that hasn't landed yet.
- All session state uses a single hardcoded `session_id` (`demo_user_001`); there's no user authentication or session isolation yet.
- `artifacts/` and `data/raw/` are git-ignored by design — model weights, the ChromaDB store, and the SQLite database are all generated or downloaded locally, never committed.

---

## ⚠️ Ethical Disclaimer

```
╔══════════════════════════════════════════════════════════════════╗
║  MindGuard is a research and portfolio demonstration project.    ║
║  It is NOT a licensed medical device and is NOT a substitute     ║
║  for professional mental health care.                            ║
║                                                                  ║
║  If you or someone you know is in crisis, please contact:        ║
║  • iCall (India):        9152987821                              ║
║  • Vandrevala Foundation: 1860-2662-345  (24x7)                  ║
║  • International:        findahelpline.com                       ║
╚══════════════════════════════════════════════════════════════════╝
```

> Helpline numbers change over time — please verify these are current before publishing or deploying this project publicly.

---

## 📄 License

This repository does not currently include a `LICENSE` file. If you plan to accept external contributions or reuse this code elsewhere, consider adding one — [MIT](https://choosealicense.com/licenses/mit/) is a common, permissive choice for projects like this.

---

## ◈ Author

<div align="center">

**Mohit Parmar**
*B.Tech CSE · DIT University, Dehradun*
*Data Science & ML Engineering*

[![GitHub](https://img.shields.io/badge/GitHub-MohitParmar78-181717?style=for-the-badge&logo=github)](https://github.com/MohitParmar78)

*Built with curiosity, caffeine, and an unreasonable belief that AI can make the world kinder.*

</div>

---

<div align="center">

*If this project helped you, a ⭐ on GitHub means more than you know.*

</div>
