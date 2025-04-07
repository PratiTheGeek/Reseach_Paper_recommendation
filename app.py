
import streamlit as st
import torch
from sentence_transformers import util
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras



embeddings = pickle.load(open('embeddings.pkl','rb'))
sentences = pickle.load(open('sentences.pkl','rb'))
rec_model = pickle.load(open('rec_model.pkl','rb'))


loaded_model = keras.models.load_model("model.h5")

with open("text_vectorizer_config.pkl", "rb") as f:
    saved_text_vectorizer_config = pickle.load(f)

loaded_text_vectorizer = TextVectorization.from_config(saved_text_vectorizer_config)

with open("text_vectorizer_weights.pkl", "rb") as f:
    weights = pickle.load(f)
    loaded_text_vectorizer.set_weights(weights)


with open("vocab.pkl", "rb") as f:
    loaded_vocab = pickle.load(f)



def recommendation(input_paper):
    # Calclllate cosine similarity scores t.
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))

    #  indices of the top-k most similar papers 
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)

    # Retrieve the titles of the top similar papers.
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])

    return papers_list



def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(loaded_vocab, hot_indices)

def predict_category(abstract, model, vectorizer, label_lookup):
    
    preprocessed_abstract = vectorizer([abstract])

   
    predictions = model.predict(preprocessed_abstract)

    
    predicted_labels = label_lookup(np.round(predictions).astype(int)[0])

    return predicted_labels


st.title('Research Papers Recommendation and Subject Area Prediction App')
st.write("Deep Learning Base App")

input_paper = st.text_input("Enter Paper title.....")
new_abstract = st.text_area("Past paper abstract....")
if st.button("Recommend"):
    # recommendation part
    recommend_papers = recommendation(input_paper)
    st.subheader("Recommended Papers")
    st.write(recommend_papers)

    #========prediction part
    st.write("===================================================================")
    predicted_categories = predict_category(new_abstract, loaded_model, loaded_text_vectorizer, invert_multi_hot)
    st.subheader("Predicted Subject area")
    st.write(predicted_categories)
