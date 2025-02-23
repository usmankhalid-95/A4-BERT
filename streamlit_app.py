import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model and tokenizer
sentence_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sentence_model = AutoModel.from_pretrained("bert-base-uncased")

# Load custom model weights
sentence_model.load_state_dict(torch.load("sbert_model.pth", map_location=torch.device("cpu")))
sentence_model.eval()

# Function to compute mean-pooled embeddings
def compute_mean_pooled_embeddings(token_embeddings, attention_mask):
    # Expand attention mask to match embedding dimensions
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Compute mean-pooled embeddings while ignoring padding tokens
    pooled_embeddings = torch.sum(token_embeddings * expanded_mask, 1) / torch.clamp(expanded_mask.sum(1), min=1e-9)
    return pooled_embeddings

# Function to compute similarity between two sentences
def compute_sentence_similarity(model, tokenizer, sentence1, sentence2):
    # Tokenize input sentences
    encoded_sentence1 = tokenizer(sentence1, return_tensors="pt", truncation=True, padding=True)
    encoded_sentence2 = tokenizer(sentence2, return_tensors="pt", truncation=True, padding=True)

    # Extract input IDs and attention masks
    input_ids1 = encoded_sentence1["input_ids"]
    attention_mask1 = encoded_sentence1["attention_mask"]
    input_ids2 = encoded_sentence2["input_ids"]
    attention_mask2 = encoded_sentence2["attention_mask"]

    # Generate token embeddings
    embeddings1 = model(input_ids1, attention_mask=attention_mask1)[0]  # Shape: (batch_size, seq_len, hidden_dim)
    embeddings2 = model(input_ids2, attention_mask=attention_mask2)[0]  # Shape: (batch_size, seq_len, hidden_dim)

    # Compute mean-pooled sentence embeddings
    pooled_embeddings1 = compute_mean_pooled_embeddings(embeddings1, attention_mask1).detach().cpu().numpy().reshape(-1)
    pooled_embeddings2 = compute_mean_pooled_embeddings(embeddings2, attention_mask2).detach().cpu().numpy().reshape(-1)

    # Calculate cosine similarity
    similarity = cosine_similarity(pooled_embeddings1.reshape(1, -1), pooled_embeddings2.reshape(1, -1))[0, 0]
    return similarity

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f7ff; /* Light gray background */
    }
    .stTextArea textarea {
        font-size: 16px;
        padding: 10px;
        height: 200px; /* Double the height of the input text windows */
        background-color: #ffffff; /* White background for text areas */
        color: #2c3e50; /* Dark blue text for input fields */
    }
    .stTextArea label {
        color: #2c3e50; /* Dark blue for input labels */
    }
    .stButton button {
        background-color: #4CAF50;
        color: #2c3e50; /* Dark blue text for button */
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #2c3e50 !important; /* Dark blue for all text */
    }
    .stMarkdown h1 {
        color: #2c3e50 !important; /* Dark blue for headings */
    }
    .stMarkdown h2 {
        color: #2c3e50 !important; /* Dark blue for subheadings */
    }
    .stMarkdown h3 {
        color: #2c3e50 !important; /* Dark blue for smaller headings */
    }
    .stMarkdown p {
        color: #2c3e50 !important; /* Dark blue for paragraph text */
    }
    .stWarning {
        color: #2c3e50 !important; /* Dark blue for warning messages */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app interface
st.title("✨ Sentence Similarity Analyzer")
st.markdown("""
    <h3 style='color: #2c3e50;'>
        Compare two sentences and analyze their similarity using advanced Sentence-BERT embeddings.
    </h3>
    """, unsafe_allow_html=True)

# Input fields for sentences
st.subheader("Enter Your Sentences")
col1, col2 = st.columns(2)
with col1:
    input_sentence1 = st.text_area("First Sentence", placeholder="Type your first sentence here...", height=200)  # Double the height
with col2:
    input_sentence2 = st.text_area("Second Sentence", placeholder="Type your second sentence here...", height=200)  # Double the height

# Button to trigger similarity calculation
if st.button("Analyze Similarity", key="analyze_button"):
    if input_sentence1 and input_sentence2:
        similarity_score = compute_sentence_similarity(sentence_model, sentence_tokenizer, input_sentence1, input_sentence2)
        st.subheader("Results")
        st.markdown(f"""
            <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);'>
                <h4 style='color: #2c3e50;'>Cosine Similarity Score: <span style='color: #4CAF50;'>{similarity_score:.4f}</span></h4>
                <h4 style='color: #2c3e50;'>Label: <span style='color: #4CAF50;'>{'Entailment' if similarity_score > 0.5 else ('Neutral' if similarity_score == 0.5 else 'Contradiction')}</span></h4>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter both sentences to analyze similarity.")