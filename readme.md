# 📄 AI Resume ATS Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and suggests concrete improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as PDF, DOCX or TXT
- Optional job description for a tailored keyword match
- Overall ATS score (0-100) plus a 5-category breakdown
- Strengths, prioritised improvements, missing keywords
- Suggested bullet-point rewrites
- Download the report as Markdown

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either:

- create `.streamlit/secrets.toml`:
  ```toml
  GEMINI_API_KEY = "your-key-here"
  ```
- or set an environment variable (`GEMINI_API_KEY`), or just paste the key into the app sidebar.

Start the app:

```bash
streamlit run app.py
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var / sidebar | required |
| `GEMINI_MODEL` | secrets / env var | `gemini-3.6-flash` |

To use a different Flash model, set `GEMINI_MODEL` (for example in secrets) without touching the code.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit `.streamlit/secrets.toml`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick your repo and branch, set the main file to `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Notes
- Text is extracted locally; scanned/image-only PDFs can't be read (and ATS systems can't read them either).
- Resumes are sent to the Gemini API for analysis. Don't upload anything you aren't comfortable sharing.
- The score is an AI estimate, not the output of a real ATS.
