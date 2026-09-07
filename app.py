import streamlit as st
import pickle

# Load trained model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Title
st.title("🎬 IMDB Sentiment Analyzer")
st.write("Enter a movie review and let the AI predict its sentiment.")

# Review input
review = st.text_area(
    "Enter your movie review:",
    height=180,
    placeholder="Example: This movie was absolutely amazing and entertaining..."
)
# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 30px;
}

.review-box {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🎬 IMDB Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered movie review sentiment classification</div>',
    unsafe_allow_html=True
)

st.divider()

# Review input
st.markdown("### 📝 Enter your movie review")

review = st.text_area(
    "",
    height=180,
    placeholder="Example: This movie was absolutely amazing and entertaining..."
)

st.caption("The model analyzes the words and phrases in your review using TF-IDF and Naive Bayes.")
# Analyze button
if st.button("🔍 Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        # Convert review into TF-IDF features
        review_tfidf = vectorizer.transform([review])

      # Predict sentiment
        prediction = model.predict(review_tfidf)[0]
        probability = model.predict_proba(review_tfidf)[0]
        confidence = max(probability) * 100
        st.info(f"🎯 Confidence: {confidence:.2f}%")
        st.progress(confidence / 100)
        st.caption(f"Model confidence: {confidence:.2f}%")
        # Display result
        if prediction == "positive":
            st.success("🟢 POSITIVE SENTIMENT")
        else:
            st.error("🔴 NEGATIVE SENTIMENT")