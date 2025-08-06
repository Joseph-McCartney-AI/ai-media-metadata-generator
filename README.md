# AI Media Tagging & Metadata Generator

This application allows you to upload an image, automatically generate a descriptive caption using the BLIP2 vision model, and then expand that caption into detailed metadata (tags, mood, and suggested usage) using GPT-4.

The tool is designed for creative and media organisations to speed up cataloguing, improve searchability, and support creative asset workflows.

---

## Features

- **Automatic Image Captioning**: Uses BLIP2 (Hugging Face Inference API) to generate an initial caption.
- **Metadata Expansion**: Refines the caption and generates:
  - Refined Caption
  - Tags (8–15 concise keywords)
  - Mood (1–3 descriptive words)
  - Suggested Usage (1–2 sentences)
- **Clean Output**: Displays results clearly and offers JSON download for integration with other systems.
- **Error Handling**: Gracefully handles missing keys, API issues, and large image uploads.

---

## Requirements

- Python 3.9+
- Hugging Face API key
- OpenAI API key

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/ai-media-tagging.git
cd ai-media-tagging
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
HUGGINGFACE_API_KEY=hf_...
OPENAI_MODEL=gpt-4o-mini
HF_BLIP2_MODEL=Salesforce/blip2-opt-2.7b
```

---

## Running Locally

```bash
streamlit run app.py
```

The app will start at `http://localhost:8501`.

---

## Deployment to Streamlit Cloud

1. Push your repository to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/) and create a new app.
3. In **App Settings**, set your repository and branch, and point to `app.py`.
4. Under **Secrets**, add:

```toml
OPENAI_API_KEY = "sk-..."
HUGGINGFACE_API_KEY = "hf-..."
OPENAI_MODEL = "gpt-4o-mini"
HF_BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
```

5. Deploy. Your app will be accessible via a public URL.

---

## Example Usage

**Input**:  
An image of a sea turtle swimming above a coral reef.

**BLIP2 Caption**:  
`underwater scene with a sea turtle swimming over coral reef`

**Generated Metadata**:

```json
{
  "refined_caption": "Sea turtle gliding above a colorful coral reef in clear tropical water.",
  "tags": ["ocean", "sea turtle", "coral reef", "marine life", "nature", "underwater", "tropical", "wildlife", "conservation", "blue palette"],
  "mood": "calm",
  "suggested_usage": "Ideal for conservation campaigns, educational materials, or ocean-themed documentaries and exhibits."
}
```

---

## Project Structure

```
ai-media-tagging/
├─ app.py
├─ README.md
├─ requirements.txt
├─ .env.example
├─ prompts/
│  └─ metadata_prompt.txt
├─ sample_images/
│  └─ example.jpg
└─ .streamlit/
   └─ secrets.toml
```

---

## Future Improvements

- Batch processing for multiple images.
- Support for video frames (extract frame → caption → metadata).
- Fine-tuning BLIP2 or using domain-specific caption models.
- Export metadata to CSV or integrate directly with DAM systems.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
