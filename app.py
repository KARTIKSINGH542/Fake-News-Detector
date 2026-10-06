"""
Fake News Detection Chatbot — Streamlit App
Serves the BERT model fine-tuned in bert_finetune.ipynb.

Run with:  streamlit run app.py
Requires the "fine_tuned_bert/" folder (produced by the notebook) to sit
next to this file.
"""

import re
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "fine_tuned_bert"
MAX_LEN = 64

st.set_page_config(page_title="Fake News Detection Chatbot", page_icon="📰", layout="centered")


# ------------------------------------------------------------------
# Load model & tokenizer once, cached across reruns/interactions
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def clean_text(text: str) -> str:
    """Same light cleaning used before tokenizing in the fine-tuning notebook —
    keep this in sync with bert_finetune.ipynb's light_clean()."""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(text: str, model, tokenizer, device):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    prediction = int(probs.argmax())
    confidence = float(probs[prediction])
    label = "REAL / DISASTER" if prediction == 1 else "NOT A REAL DISASTER (likely fake/unrelated)"
    return label, confidence, prediction


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📰 Fake News Detection Chatbot")
st.caption(
    "Paste a news snippet or tweet below. The chatbot uses a fine-tuned BERT model "
    "to classify it and reply with a prediction and confidence score."
)

try:
    model, tokenizer, device = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        f"Couldn't load the model from '{MODEL_DIR}/'. Make sure you've run "
        f"bert_finetune.ipynb through Step 12 (Save the Fine-Tuned Model) and that the "
        f"'{MODEL_DIR}' folder sits next to app.py.\n\nError: {e}"
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Paste a news snippet or tweet and I'll tell you if it looks fake or real.",
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if model_loaded:
    user_input = st.chat_input("Type or paste a news article / tweet snippet...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        label, confidence, prediction = predict(user_input, model, tokenizer, device)
        confidence_pct = round(confidence * 100, 2)

        if prediction == 1:
            reply = f"🟢 This looks like **real / disaster-related** news, with a confidence of **{confidence_pct}%**."
        else:
            reply = f"🔴 This looks **fake / not disaster-related**, with a confidence of **{confidence_pct}%**."

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
            st.progress(confidence)
else:
    st.info("Fix the model loading error above, then reload this page.")

with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot classifies input text as real/disaster-related or fake/unrelated, "
        "using a BERT model fine-tuned for binary classification."
    )
    st.write("Model directory:", f"`{MODEL_DIR}`")
    if model_loaded:
        st.success(f"Model loaded on: {device}")
    if st.button("Clear conversation"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! Paste a news snippet or tweet and I'll tell you if it looks fake or real.",
            }
        ]
        st.rerun()