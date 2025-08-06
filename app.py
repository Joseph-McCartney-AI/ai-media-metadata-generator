# app.py
import io
import json
import os
from typing import Tuple, Optional, Dict, Any

import requests
import streamlit as st
from PIL import Image, ImageOps
from dotenv import load_dotenv

# OpenAI client (v1.x)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # handled below


# -------------------------------
# Configuration & Secrets Loading
# -------------------------------
load_dotenv()  # local .env

# Prefer Streamlit secrets in cloud, fallback to env vars locally
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY", ""))
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
HF_BLIP2_MODEL = st.secrets.get("HF_BLIP2_MODEL", os.getenv("HF_BLIP2_MODEL", "Salesforce/blip2-opt-2.7b"))

# Validate availability of OpenAI client
if OpenAI is None:
    st.error("OpenAI SDK not available. Check your requirements.txt matches the README.")
    st.stop()

# Instantiate OpenAI client
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# Load prompt from file
PROMPT_PATH = os.path.join("prompts", "metadata_prompt.txt")
if os.path.exists(PROMPT_PATH):
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        METADATA_PROMPT = f.read()
else:
    METADATA_PROMPT = (
        "Given a raw image caption, return ONLY valid JSON with keys: "
        '{"refined_caption": string, "tags": [string], "mood": string, "suggested_usage": string}. '
        "Keep tags concise and relevant."
    )


# -------------------------------
# Helper Functions
# -------------------------------
def downscale_image(img: Image.Image, max_side: int = 1280) -> Image.Image:
    """Downscale very large images to cut upload time and API latency."""
    img = ImageOps.exif_transpose(img)  # correct orientation
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.read()


def query_hf_blip2_caption(image_bytes: bytes, model_id: str, api_key: str) -> Tuple[Optional[str], Optional[str]]:
    """Calls Hugging Face Inference API for image captioning."""
    if not api_key:
        return None, "Missing Hugging Face API key."
    if not model_id:
        return None, "Missing Hugging Face model id."

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, data=image_bytes, timeout=60)
        if resp.status_code == 503:
            return None, "Model is loading on Hugging Face (503). Please try again shortly."
        if resp.status_code >= 400:
            return None, f"Hugging Face API error {resp.status_code}: {resp.text}"

        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            return None, f"Hugging Face API error: {data['error']}"

        caption = None
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            caption = first.get("generated_text") or first.get("caption") or None

        if not caption:
            return None, "No caption generated from BLIP2."
        return caption.strip(), None

    except requests.exceptions.RequestException as e:
        return None, f"Network error calling Hugging Face API: {e}"
    except Exception as e:
        return None, f"Unexpected error calling Hugging Face API: {e}"


def parse_json_safe(text: str) -> Optional[Dict[str, Any]]:
    """Try to locate and parse a JSON object in a model's string output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def call_openai_for_metadata(caption: str, model: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Sends the caption to OpenAI to get structured metadata."""
    try:
        messages = [
            {"role": "system", "content": METADATA_PROMPT},
            {"role": "user", "content": f"Raw image caption:\n\"{caption}\""},
        ]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        raw = completion.choices[0].message.content or ""
        data = parse_json_safe(raw)
        if not data:
            return None, raw, "Failed to parse JSON from OpenAI response."
        for key in ["refined_caption", "tags", "mood", "suggested_usage"]:
            if key not in data:
                return None, raw, f"Missing key in JSON: {key}"
        if not isinstance(data.get("tags"), list):
            return None, raw, "Expected 'tags' to be a list."
        return data, raw, None

    except Exception as e:
        return None, None, f"OpenAI API error: {e}"


def key_check() -> bool:
    """Check required API keys exist."""
    missing = []
    if not HUGGINGFACE_API_KEY:
        missing.append("HUGGINGFACE_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        st.warning(
            "Missing required secrets: " + ", ".join(missing) +
            ". Add them to `.env` for local dev or `.streamlit/secrets.toml` for Streamlit Cloud."
        )
        return False
    return True


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="AI Media Tagging & Metadata Generator",
    layout="centered",
)

st.title("AI Media Tagging & Metadata Generator")
st.caption("Upload an image → BLIP2 caption → GPT-4 metadata (tags, mood, suggested usage).")
with st.expander("About this tool"):
    st.markdown(
        """
**What it does**
- Generates a descriptive caption from your image using BLIP2 (Hugging Face).
- Expands that caption into useful media metadata via GPT-4:
  - Refined Caption
  - Tags (8–15 concise keywords)
  - Mood (1–3 words)
  - Suggested Usage (1–2 sentences)

**Why it’s useful**
- Speeds up cataloguing, search, and creative workflows in media teams.
- Produces consistent, scannable metadata for DAMs and archives.
        """
    )

uploaded = st.file_uploader(
    "Upload an image (PNG or JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

with st.sidebar:
    st.header("Settings")
    st.text_input("Hugging Face Model", value=HF_BLIP2_MODEL, key="hf_model")
    st.text_input("OpenAI Model", value=OPENAI_MODEL, key="oa_model")
    st.write("---")
    st.caption("Tip: Keep images under ~3000px on the long side for faster processing.")

if uploaded is not None:
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Failed to open image. Please upload a valid PNG/JPEG.")
        st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Preview")
        st.image(img, use_column_width=True)

    with col2:
        if not key_check():
            st.stop()

        img_small = downscale_image(img, max_side=1600)
        img_bytes = image_to_bytes(img_small, format="PNG")

        if st.button("Generate Metadata", type="primary"):
            with st.status("Generating caption with BLIP2…", expanded=False):
                caption, hf_err = query_hf_blip2_caption(
                    image_bytes=img_bytes,
                    model_id=st.session_state["hf_model"].strip(),
                    api_key=HUGGINGFACE_API_KEY,
                )

            if hf_err:
                st.error(hf_err)
                st.stop()

            st.success("Caption generated")
            st.write(f"BLIP2 Caption: {caption}")

            with st.status("Expanding to metadata with GPT-4…", expanded=False):
                meta, raw_text, oa_err = call_openai_for_metadata(
                    caption=caption, model=st.session_state["oa_model"].strip()
                )

            if oa_err:
                st.error(oa_err)
                if raw_text:
                    with st.expander("See raw model output"):
                        st.code(raw_text, language="json")
                st.stop()

            st.subheader("Results")
            st.markdown(f"Refined Caption: {meta['refined_caption']}")
            st.markdown("Tags:")
            st.write(", ".join(meta["tags"]))
            st.markdown(f"Mood: {meta['mood']}")
            st.markdown(f"Suggested Usage: {meta['suggested_usage']}")

            result_json = json.dumps(meta, ensure_ascii=False, indent=2)
            st.download_button(
                label="Download JSON",
                data=result_json.encode("utf-8"),
                file_name="metadata.json",
                mime="application/json",
            )

            with st.expander("View JSON"):
                st.code(result_json, language="json")

else:
    st.info("Upload an image to get started.")
