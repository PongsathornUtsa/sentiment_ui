import streamlit as st
import torch
import torch.nn.functional as F
from pythainlp.tokenize import word_tokenize
import plotly.graph_objects as go


def render_story_tab(tokenizer, model):
    st.subheader("🚀 **WangchanBERTa Sentiment Analyzer**")
    st.write("🔍 วิเคราะห์ความคิดเห็นจากข้อความภาษาไทย")

    id2label = {
        0: "positive",
        1: "neutral",
        2: "negative",
    }
    example = "พนักงานบริการดีมาก สัญญาณก็ดี แต่ร้านอยู่ที่ไหน อยากได้ข้อมูลเพิ่มเติม จะได้ประกาศบนเว็บถูก"

    with st.form(key="sentiment_analysis_form"):
        input_text = st.text_area("Enter text for sentiment analysis:", value=example)
        submit_button = st.form_submit_button(label="Analyze", type="primary")

    if submit_button:
        if input_text:
            segmented_text = word_tokenize(input_text, engine="longest")
            preprocessed_text = " ".join(segmented_text)

            st.write("**Tokenized Words:**")

            fixed_color = "#ff4b4b"
            colored_tokens = [
                f'<span style="background-color: {fixed_color};color: white; padding: 1px 6px; '
                "margin: 0 5px; display: inline; vertical-align: middle; "
                "border-radius: 0.25rem; font-size: 1rem; font-weight: 400; "
                f'white-space: nowrap">{token}</span>'
                for token in segmented_text
                if token.strip() != ""
            ]

            st.markdown(
                f'<div style="word-wrap: break-word;">{"".join(colored_tokens)}</div>',
                unsafe_allow_html=True,
            )

            inputs = tokenizer(
                preprocessed_text, return_tensors="pt", padding=True, truncation=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            probs = F.softmax(logits, dim=-1)

            prob_negative = probs[0][2].item() * 100
            prob_neutral = probs[0][1].item() * 100
            prob_positive = probs[0][0].item() * 100

            predicted_class = torch.argmax(probs, dim=-1).item()
            predicted_label = id2label[predicted_class]

            st.write("")
            st.write(f"**Predicted Sentiment:** **{predicted_label}**")

            labels = ["Positive", "Neutral", "Negative"]
            confidence_levels = [prob_positive, prob_neutral, prob_negative]

            fig = go.Figure(
                [
                    go.Bar(
                        x=labels,
                        y=confidence_levels,
                        marker_color=["green", "blue", "red"],
                    )
                ]
            )
            fig.update_layout(
                title="Confidence Levels for Sentiment Prediction",
                xaxis_title="Sentiment",
                yaxis_title="Confidence (%)",
                yaxis_range=[0, 100],
            )

            st.plotly_chart(fig)

        else:
            st.warning("Please enter some text to analyze.")
