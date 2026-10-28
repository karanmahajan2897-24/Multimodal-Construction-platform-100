# Multimodal Construction Intelligence Platform
Skill Wallet Capstone Project 1: a single-file Streamlit app with no database.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to a GitHub repository.
2. On share.streamlit.io choose **New app**, select the repo, and set the main file to `app.py`.
3. Optional: under **Advanced settings > Secrets** add `GEMINI_API_KEY = "your-key"` so visitors get live mode by default. Without it, users paste a key in the sidebar or use demonstration mode.

## Modes
- **Live mode**: a Gemini key is present. Photos are analysed by the selected Gemini model and results are validated before display.
- **Demonstration mode**: no key. Every module returns realistic sample output so the full workflow can be shown offline.

## Notes
- Default model is `gemini-2.5-flash`. Change it in the sidebar, or pick "Custom…" to enter any model ID your key can use.
- Valuation rates in `STRUCTURE_TYPES` and `LOCATION_TIERS` are indicative averages. Edit them in `app.py` or use the assumptions panel in the app.
- Outputs are decision-support aids, not engineering certifications.
