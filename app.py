import streamlit as st
import pickle
import joblib
import torch
import numpy as np
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sentence_transformers import SentenceTransformer
import os
os.environ["STREAMLIT_WATCH_MODE"] = "false"


class TransformerEmbedding:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        return self.embedder.encode(X.tolist(), convert_to_numpy=True, show_progress_bar=False)
    
# Configuration
st.set_page_config(
    page_title="✨ News Classification Platform",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* (Keep your existing custom CSS styles here) */
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'submitted_data' not in st.session_state:
    st.session_state.submitted_data = None

# Header
st.image("https://images.unsplash.com/photo-1620712943543-bcc4688e7485?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&h=400", 
         use_container_width=True, 
         caption="News Classification System")

# Title
st.markdown('<p class="title-text">📰 News Article Classification</p>', unsafe_allow_html=True)

# Main layout
with st.container():
    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container(border=True):
            st.markdown("### 📋 Input Parameters")

            model_option = st.selectbox(
                "*Select Classification Model*",
                [ "LOGISTIC REGRESSION", "TRANSFORMER CLASSIFIER"],
                index=0
            )

            headline = st.text_input(
                "*Article Headline*",
                placeholder="Enter news headline...",
                value=""
            )

            description = st.text_area(
                "*Article Description*",
                placeholder="Enter news description...",
                height=100,
                value=""
            )

            submit = st.button("🔍 Classify Article", type="primary", use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("### 📊 Classification Results")

            if submit or st.session_state.submitted_data:
                if not headline or not description:
                    st.warning("⚠ Please fill both headline and description fields!")
                else:
                    # Combine text inputs
                    combined_text = f"{headline} {description}".strip()
                    
                    try:
                        prediction = None
                        confidence = None
                        
                        

                        if model_option == "LOGISTIC REGRESSION":
                            # Logistic Regression prediction
                            with open('multiclass_logistic_regression.pkl', 'rb') as f:
                                pipeline = pickle.load(f)
                            le = joblib.load('./results/best_model/label_encoder.pkl')
                            
                            prediction = pipeline.predict([combined_text])[0]
                            confidence = np.max(pipeline.predict_proba([combined_text]))
                            prediction = le.inverse_transform([prediction])[0]

                        elif model_option == "TRANSFORMER CLASSIFIER":
                            # Transformer prediction
                            model = DistilBertForSequenceClassification.from_pretrained("./results/best_model")
                            tokenizer = DistilBertTokenizer.from_pretrained("./results/best_model")
                            label_encoder = joblib.load('./results/best_model/label_encoder.pkl')
                            
                            inputs = tokenizer(combined_text, return_tensors="pt", 
                                            padding=True, truncation=True, max_length=128)
                            model.eval()
                            with torch.no_grad():
                                outputs = model(**inputs)
                                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                                confidence = torch.max(probs).item()
                                prediction = torch.argmax(probs).item()
                            
                            prediction = label_encoder.inverse_transform([prediction])[0]

                        # Display results
                        st.success("✔ Classification Completed!")
                        
                        with st.container(border=True):
                            st.markdown(f"#### 📰 Article Preview")
                            st.write(f"**Headline:** {headline}")
                            st.write(f"**Description:** {description[:200]}...")
                        
                        cols = st.columns(2)
                        cols[0].metric("Predicted Category", prediction)
                        cols[1].metric("Confidence Score", f"{confidence:.2%}")
                        
                        st.markdown("### 📈 Prediction Details")
                        st.write(f"**Selected Model:** {model_option}")
                        st.write(f"**Text Length:** {len(combined_text)} characters")
                        st.write(f"**Processing Time:** Instantaneous")

                    except Exception as e:
                        st.error(f"❌ Classification Error: {str(e)}")

st.divider()
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666; font-size: 0.9rem;">
    <p>© 2024 News Classification System | 
    <a href="#" style="color: #6e8efb; text-decoration: none;">Privacy Policy</a></p>
</div>
""", unsafe_allow_html=True)