# ResumeAnalyzer

> 🚀 A minimal, free-first Resume Analyzer that parses your resume, ingests a job description, scores relevance, highlights gaps, and generates tailored resume variants. Built with Streamlit + local LLMs (Ollama).

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-red">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>


## ✨ Features (MVP)
- **Upload resume** (PDF/DOCX) → robust text extraction
- **JD ingest** by URL or paste → automatic clean-up
- **Relevance scoring** via sentence-transformer embeddings
- **Skills coverage** (heuristic) using a seed skills list
- **LLM analysis** (via **Ollama**, free/local) for:
  - Actionable edits & gap analysis
  - Tailored bullet rewrites
- **Resume generation** → Markdown and downloadable **.docx**

> Works offline for most steps. LLM features auto-enable if Ollama is running locally.

---

## 🧱 Tech Stack
- **UI:** Streamlit
- **Resume Parsing:** `pdfplumber` (PDF), `python-docx` (DOCX)
- **NLP:**
  - Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`)
  - NER/Cleanup: spaCy `en_core_web_sm` (optional)
  - Zero-shot labels (later): `facebook/bart-large-mnli`
- **Similarity / RAG:** FAISS (or Chroma later)
- **LLM (free/local):** Ollama (`llama3.1:8b` or `mistral`)  
- **Prompting:** Direct prompts (LangChain optional)
- **JD Fetch:** `requests` + `BeautifulSoup`
- **Storage:** Filesystem (upgrade to SQLite later)
- **Doc generation:** Markdown + `python-docx` (PDF via WeasyPrint later)

---

## 📦 Project Structure

```text
ResumeAnalyzer/
├─ app.py # Streamlit app (upload → scrape → analyze → generate)
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ .env.example
├─ data/
│ └─ skill_dict.json # seed skills list for coverage checks
├─ outputs/ # generated resumes
│ └─ .gitkeep
└─ .streamlit/
└─ config.toml
```


## 🚀 Quickstart

### 1) Create a clean environment (recommended)
```bash
conda create -n resumeanalyzer python=3.11 -y
conda activate resumeanalyzer
```
### 2) Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
conda install -y -c conda-forge numpy
```
- installs spaCy’s small English model into your current Python environment so you can do nlp = spacy.load("en_core_web_sm")

### 3) Run LLM and UI
```bash
ollama pull llama3.1:8b
streamlit run app.py
```
- 

## 🔍 How It Works (MVP Flow)
- Parse resume → PDF/DOCX → raw text → split into bullets
- Fetch/clean JD from URL or pasted text
- Embeddings (all-MiniLM-L6-v2) → relevance score for top bullets
- Skills coverage (simple keyword match using data/skill_dict.json)
- LLM analysis (if Ollama available) → gap list + bullet rewrites
- Tailored resume → Markdown → export .docx and .md

## 🧭 Roadmap / Nice-to-Haves
- Structured parsing of roles/dates/education with spaCy patterns
- Skill taxonomy & fuzzy matching (e.g., O*NET/ESCO)
- Multi-variant generation (ATS-plain / Impact-first / One-page strict)
- Cover letter auto-draft from tailored resume
- Template-aware output (use a provided PDF/DOCX layout)
- RAG for section-to-section alignment (JD ↔ resume)
- Overall “JD match” score with weighted skills/experience
- Session/version persistence (SQLite)
- Diff view for bullets + “why it helps” tooltips

## 📚 Glossary
- **JD**: Job Description — the role posting you’re targeting.
- **ATS**: Applicant Tracking System — software that parses and filters resumes.

## 🔐 Privacy
- Local-first by default. If you later enable external APIs, store keys in .env and don’t commit them.

## 🤝 Contributing
- PRs welcome. Open an issue to discuss significant changes.
