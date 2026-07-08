# Fully-public deploy on Hugging Face Spaces (no visitor login)

Streamlit Community Cloud currently sends visitors to a sign-in page. **Hugging Face Spaces** serves
public apps with **no login for viewers** — the right home for a portfolio demo. The app still
auto-updates its data (live fetch + 24h cache), so there's nothing to maintain after setup.

## Simpler option first
If Streamlit Community Cloud lets you make the app public — open the app at share.streamlit.io →
**Settings → Sharing → "Public"** — that's a one-click fix and you can skip Hugging Face. Use the steps
below only if a fully-public (no login) option isn't available there.

## Hugging Face Spaces (one time, ~5 min; needs a free Hugging Face account)

1. **Create a write token:** https://huggingface.co/settings/tokens → *New token* → role **write**. Copy it.

2. **Create the Space:** https://huggingface.co/new-space
   - Space name: **solar-forecast**
   - **SDK: Streamlit**  ·  Hardware: **CPU basic** (free)  ·  Visibility: **Public**
   - This creates a Space whose `README.md` already carries the Streamlit config header — leave that file as is.

3. **Push the app files into the Space** (PowerShell, run from `C:\Projects`):
   ```powershell
   git clone https://huggingface.co/spaces/<your-hf-username>/solar-forecast hf-space
   Copy-Item -Recurse -Force `
     C:\Projects\solar-forecast\app.py, `
     C:\Projects\solar-forecast\requirements.txt, `
     C:\Projects\solar-forecast\src `
     hf-space\
   cd hf-space
   git add app.py requirements.txt src
   git commit -m "Add solar-forecast app"
   git push
   ```
   When prompted: username = your HF username, password = the **write token** from step 1.

The Space builds automatically and gives a public URL like
`https://huggingface.co/spaces/<your-hf-username>/solar-forecast` — open to anyone, any device, no login.

## Notes
- Uses the same `requirements.txt` (`streamlit` + `streamlit-geolocation`) and the `src/` import shim in `app.py`.
- **Data auto-renews** via the live fetch + 24h cache; no commits needed to stay current.
- To update the code later: re-copy the files into `hf-space` and `git push` again.
