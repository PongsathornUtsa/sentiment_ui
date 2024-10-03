import streamlit as st
import logging
from app_tab1 import render_story_tab
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)


@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(
        "Pongsathorn/wangchanberta-base-sentiment"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "Pongsathorn/wangchanberta-base-sentiment"
    )
    return tokenizer, model


st.header("Sentiment Analysis Model", divider="rainbow")

tokenizer, model = load_models()

tab1, tab2 = st.tabs(["Analyze", "Next Tab"])

with tab1:
    render_story_tab(tokenizer, model)
