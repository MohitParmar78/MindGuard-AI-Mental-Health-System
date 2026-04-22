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
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-XLM--RoBERTa-FFD21E?style=for-the-badge)](https://huggingface.co)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3-F55036?style=for-the-badge)](https://groq.com)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-008080?style=for-the-badge)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br>

> **MindGuard** is a production-grade, multilingual mental health AI that doesn't just respond —  
> it *diagnoses*, *explains*, and *alerts*. Every word you type is analyzed by a fine-tuned  
> XLM-RoBERTa neural network, explained by Game Theory mathematics (SHAP),  
> and answered by a clinical-context-aware Groq LLM.

</div>

---

<br>

### 🏗️ Core Architecture:
* **The Brain (Inference):** A fine-tuned XLM-RoBERTa model classifying 35 distinct emotional states and determining clinical risk levels (High/Medium/Low).
* **The Logic (Custom RAG):** A custom-built Retrieval-Augmented Generation pipeline utilizing ChromaDB to fetch verified psychological coping strategies based on the live diagnosis.
* **The Voice (LLM Generation):** Groq API integration (Llama-3.3-70b) seamlessly weaves the retrieved context into conversational empathy.
* **The Proof (XAI):** Real-time SHAP integration generating interactive HTML visualizations to prove the mathematical weight of specific words on the model's prediction.
* **The Interface:** A responsive Streamlit dashboard featuring Whisper-powered audio input, real-time metric tracking, and decoupled Hugging Face Model Hub weight loading.

## ◈ What Makes This Different

Most mental health chatbots are wrappers around GPT. **MindGuard is not.**

| Capability | Generic Chatbot | **MindGuard** |
|---|---|---|
| Response generation | ✅ GPT/Claude API call | ✅ Groq LLaMA 70 with clinical system prompt |
| Emotion detection | ❌ Guessed from LLM output | ✅ Dedicated XLM-RoBERTa (35 emotions, fine-tuned) |
| Why did it predict that? | ❌ Black box | ✅ SHAP word-level attribution — mathematically proven |
| Risk escalation | ❌ None | ✅ High / Medium / Low triage with visual alerts |
| Multilingual | ❌ English only | ✅ 100+ languages via XLM-R architecture |
| Voice input | ❌ None | ✅ OpenAI Whisper transcription pipeline |
| Clinical audit trail | ❌ None | ✅ Full SQLite session history with timestamps |
| XAI dashboard | ❌ None | ✅ Clinician-facing SHAP HTML report viewer |

<br>

---

## ◈ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                         │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐   │
│  │    💬 Chat Companion      │    │   📊 Clinical Dashboard       │   │
│  │  • Text input            │    │  • Emotion frequency chart    │   │
│  │  • Voice recording       │    │  • Risk distribution chart    │   │
│  │  • Emotion badge display │    │  • Session history table      │   │
│  │  • Risk badge display    │    │  • 🔬 SHAP report viewer      │   │
│  │  • Inline SHAP report    │    │                               │   │
│  └────────────┬─────────────┘    └──────────────────────────────┘   │
└───────────────┼──────────────────────────────────────────────────────┘
                │ user input (text / audio)
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         PROCESSING LAYER                             │
│                                                                       │
│   ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│   │   Whisper   │───▶│  XLM-RoBERTa     │───▶│   SHAP Engine    │   │
│   │  (audio →  │    │  Emotion Model   │    │  (Game Theory    │   │
│   │   text)    │    │  35 categories   │    │   attribution)   │   │
│   └─────────────┘    └────────┬─────────┘    └──────────────────┘   │
│                               │ emotion + risk                        │
│                               ▼                                       │
│                    ┌──────────────────────┐                          │
│                    │   Groq LLaMA 3 LLM   │                          │
│                    │  (clinical response  │                          │
│                    │   generation)        │                          │
│                    └──────────┬───────────┘                          │
└───────────────────────────────┼──────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          PERSISTENCE LAYER                           │
│   SQLite DB (session history)    artifacts/shap_report.html          │
└─────────────────────────────────────────────────────────────────────┘
```

<br>

---

## ◈ The XAI Engine — Why This Matters

> *"Trust, but verify."* — Every prediction MindGuard makes can be traced back to the exact words that caused it.

MindGuard uses **SHAP (SHapley Additive exPlanations)** — a mathematically rigorous framework rooted in cooperative Game Theory — to answer:

**"Which specific words made the model predict *Anxiety* over *Depression*?"**

```
Input:  "I have a massive presentation tomorrow and my chest is tight."
                │                    │                        │
                ▼                    ▼                        ▼
           [neutral]           [HIGH IMPACT]            [HIGH IMPACT]
           SHAP ≈ 0.01         "presentation"           "chest is tight"
                               SHAP = +0.43              SHAP = +0.51
                                    │                        │
                                    └──────────┬─────────────┘
                                               ▼
                                    Predicted: ANXIETY (87.3%)
                                    Risk Level: MEDIUM ⚠️
```

This is rendered as an **interactive HTML report** embedded directly in the chat — red highlights push the prediction *toward* the emotion, blue highlights push *against* it.

<br>

---

## ◈ Quick Start

### Prerequisites

```bash
python >= 3.10
```

### 1 · Clone & Install

```bash
git clone https://github.com/MohitParmar78/MindGuard-AI-Mental-Health-System.git
cd MindGuard-AI-Mental-Health-System
pip install -r requirements.txt
```

### 2 · Configure Secrets

Create a `.env` file at the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 3 · Ensure Model Weights Exist

The XLM-RoBERTa fine-tuned weights must be present at:

```
artifacts/
└── xlmr_weights/
    └── final_mindguard_model/
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer_config.json
        └── vocab.json
```

> Train your own using the notebooks in `notebooks/` or download pretrained weights.

### 4 · Launch

```bash
cd app
streamlit run app/main.py
```

Navigate to `http://localhost:8501` in your browser.

<br>

---

## ◈ Project Structure

```
MindGuard-AI-Mental-Health-System/
│
├── 📁 app/                          ← Streamlit application
│   ├── main.py                      ← Entry point, page routing
│   ├── api.py                       ← Cached model loaders (@st.cache_resource)
│   └── components/
│       ├── chat_ui.py               ← Chat interface + SHAP + badge rendering
│       └── dashboard_ui.py          ← Analytics + XAI report tab
│
├── 📁 src/                          ← Core business logic
│   ├── chatbot/
│   │   └── groq_bot.py              ← Groq LLM orchestration + DB writes
│   ├── explainability/
│   │   └── shap_explainer.py        ← XLM-R + SHAP pipeline → HTML report
│   ├── database/
│   │   └── db_operations.py         ← SQLite CRUD operations
│   ├── audio/
│   │   └── audio_processor.py       ← Whisper transcription wrapper
│   ├── preprocessing/               ← Text cleaning and formatting
│   │   └── cleaner.py               ← Sanitization logic for inputs/documents
│   └── rag_engine/                  ← Retrieval-Augmented Generation logic
│       ├── build_vectordb.py        ← Vector database ingestion/creation
│       └── retriever.py             ← Semantic search and context fetching
│
├── 📁 artifacts/                    ← Model weights + generated reports
│   ├── xlmr_weights/
│   │   └── final_mindguard_model/   ← Fine-tuned XLM-RoBERTa (35 emotions)
│   └── shap_report.html             ← Latest SHAP explanation (auto-generated)
│
├── 📁 data/
│   └── raw/                         ← Temp audio files from voice input
│
├── 📁 notebooks/                    ← Training + experimentation notebooks
├── requirements.txt
└── README.md
```

<br>

---

## ◈ Emotion & Risk Classification

### 35-Class Emotion Taxonomy

MindGuard classifies inputs across two merged ontologies:

**Clinical Diagnoses** (7 classes)

| Label | Severity | Risk Mapping |
|---|---|---|
| 🔴 Suicidal | Critical | → **HIGH** |
| 🔴 Depression | Severe | → **HIGH** |
| 🟠 Anxiety | Moderate–Severe | → **MEDIUM–HIGH** |
| 🟠 Bipolar | Moderate–Severe | → **MEDIUM–HIGH** |
| 🟠 Stress | Moderate | → **MEDIUM** |
| 🟡 Personality Disorder | Varies | → **MEDIUM** |
| 🟢 Normal | None | → **LOW** |

**GoEmotions Fine-Grained** (28 classes) — admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, neutral, optimism, pride, realization, relief, remorse, sadness, surprise

<br>

---

## ◈ Technology Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Groq + LLaMA 3 | Ultra-low latency inference (<1s), free tier available |
| **Emotion Model** | XLM-RoBERTa (fine-tuned) | Multilingual, 100+ languages, state-of-art on NLP benchmarks |
| **Explainability** | SHAP `shap.Explainer` | Only mathematically rigorous word attribution framework |
| **Speech-to-Text** | OpenAI Whisper | Best-in-class accuracy, runs fully offline |
| **UI Framework** | Streamlit | Rapid ML app development, Python-native |
| **Database** | SQLite | Zero-config, portable, sufficient for session-scale data |
| **ML Utilities** | PyTorch, Transformers | Industry standard deep learning stack |

<br>

---

## ◈ Roadmap

- [x] XLM-RoBERTa 35-emotion fine-tuned classifier
- [x] SHAP word-level explainability with HTML report
- [x] Groq LLM integration with clinical system prompt
- [x] Whisper voice-to-text pipeline
- [x] SQLite audit trail + clinical dashboard
- [x] Real-time risk badge system (High / Medium / Low)
- [ ] RAG integration — retrieve clinical CBT strategies from knowledge base
- [ ] Multi-session memory with persistent user profiles
- [ ] Longitudinal mood tracking with trend analysis

<br>

---

## ◈ Ethical Disclaimer

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

<br>

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
