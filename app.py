import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import os
from io import BytesIO

import nltk

# Ensure the 'wordnet' resource is downloaded
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd

# base_path = os.getcwd()
sub_path = 'data'
dataset_path = os.path.join(sub_path, 'mobile_recommendation_system_dataset.csv')

# import df
df = pd.read_csv(dataset_path,index_col=False)
def recommend(mobile):
    mobile_index = df[df['name']==mobile].index[0]
    similarity_array = similarity[mobile_index]
    similar_10_mobiles = sorted(list(enumerate(similarity_array)),reverse=True,key=lambda x:x[1])[1:11]
    return similar_10_mobiles


def avg_word2vec(doc):
    return np.mean([name_model.wv[word] for word in doc if word in name_model.wv.index_to_key],axis=0)

# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="PickYourPhone", page_icon=":mobile_phone:", layout="wide")



def load_lottiefile(filepath):
    import json
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# ---- LOAD ASSETS ----
sub_dir = os.path.join(os.getcwd(), 'animation_files')
phoneThinkingANIMI = load_lottiefile(os.path.join(sub_dir ,"Animation - 1733853644589.json"))
phones = load_lottiefile(os.path.join(sub_dir ,"Animation - 1733047055915.json"))
coding = load_lottiefile(os.path.join(sub_dir ,"Animation - 1733854130756.json"))

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")


# ---- HEADER SECTION ----
with st.container():
    st.header("Hi, I am Shyam Goli :wave:")
    left_column, right_column = st.columns(2)
    with left_column:
        st_lottie(coding, height=300, key="coding")
    with right_column:
        st.title("A Python AI/ML engineer from India")
        st.write("I am a Python enthusiast with a keen interest in machine learning and artificial intelligence. I've developed several projects leveraging technologies such as ML, NLP, Computer Vision and Gen AI. In addition to my backend expertise, My programming experience spans C, C++, and Python. I currently work at Capgemini in Mumbai as an Senior Associate Software Engineer, specializing in Python and AI/ML and started my journey of study about Gen AI. I did some Google Data Analytics certifications. I have a passion for exploring emerging technologies related to AI and continuously updating my skill set.")
        st.write("[LinkedIn Profile >](https://www.linkedin.com/in/shyam-goli-723657176/)")
    

# ---- WHAT this project does explaination ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("About project")
        st.write("##")
        st.write(
            """
            I developed a recommendation system for mobile phones utilizing NLP techniques.
            - The system employs Word2Vec to generate word embeddings and cosine similarity to identify the most relevant recommendations.
            - A user-friendly GUI was created using Streamlit, allowing users to explore personalized suggestions effectively. 
            - This system helps users quickly find mobile phones that match their preferences,saving time and enhancing decision-making by providing tailored recommendations based on their requirements.
            - This recommendation system can be upgraded by using updated embedding methods.

            If this sounds interesting to you, consider subscribing and turning on the notifications, so you don’t miss any content.
            """
        )
        st.write("[Github Code Repo >](https://github.com/GOLISHYAMP/Recommendation_System.git)")
    with right_column:
        st_lottie(phoneThinkingANIMI, height=300, key="thinking")
        st_lottie(phones, height=300, key="phones")

# ---- Recommendations ----
import pickle
# Load the saved model
model_path = 'src'
model_subpath= 'model'
with open(os.path.join(model_path, model_subpath, 'similarity.pkl'), 'rb') as file:
    similarity = pickle.load(file)

with open(os.path.join(model_path, model_subpath, 'name_model.pkl'), 'rb') as file:
    name_model = pickle.load(file)

with open(os.path.join(model_path, model_subpath, 'name_vectors.pkl'), 'rb') as file:
    name_vectors = pickle.load(file)




st.write("---")
st.header("Mobile Recommendation")
st.write("##")

# Set the title
st.title("Search for your mobile")
import time
# Add a text input with a search icon as a placeholder
input_string = st.text_input("Search", placeholder="🔍 Type your query here...", key='input_string')

similar_10_mobiles = None
# Function to handle card selection
def handle_card_selection(card_text):
    global similar_10_mobiles, input_string
    # st.session_state.selected_card = card_text
    # print(card_text)
    # input_string = 
    similar_10_mobiles = recommend(card_text)
    clear_search()
    # Initialize session state
    if "input_string" not in st.session_state:
        st.session_state.input_string = ""
    st.session_state.input_string = card_text
    st.rerun()

# Function to clear the input field
def clear_search():
    del st.session_state.input_string

# Initialize session state
if "input_string" not in st.session_state:
    st.session_state.input_string = ""


# Display the entered search query
if st.session_state.input_string:
    st.write(f"You searched for: {st.session_state.input_string}")
    if st.session_state.input_string in list(df['name']):
        st.write(df.loc[df['name'] == st.session_state.input_string, 'corpus'].iloc[0])
    st.header("Recommendations")
    input_string = st.session_state.input_string.lower()
    input_string = input_string.split()
    input_string = [lemmatizer.lemmatize(word) for word in input_string]
    input_string = ' '.join(input_string)
    # print(input_string)
    sent_token=sent_tokenize(input_string)
    # print(sent_token)
    sim_pro = word_tokenize(sent_token[0])
    # print(sim_pro)
    w2v = avg_word2vec(sim_pro)
    # print(w2v)

    w2v_reshaped = w2v.reshape(1,-1)
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_name_words = cosine_similarity(w2v_reshaped, name_vectors)

    similar_10_mobiles = sorted(list(enumerate(similarity_name_words[0])),reverse=True,key=lambda x:x[1])[0:10]

    with st.container():
        for i in similar_10_mobiles:
            title_column, text_column = st.columns((1, 2))
            with title_column:
                # st.title()
                if st.button(df['name'].iloc[i[0]]):
                    handle_card_selection(df['name'].iloc[i[0]])
            with text_column:
                st.subheader(df['ratings'].iloc[i[0]], df['price'].iloc[i[0]])
                st.write(
                    df['corpus'].iloc[i[0]]
                )
          #########################################

# with st.container():
#     st.write("---")
#     st.header("Get In Touch With Me!")
#     st.write("##")

#     # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
#     contact_form = """
#     <form action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
#         <input type="hidden" name="_captcha" value="false">
#         <input type="text" name="name" placeholder="Your name" required>
#         <input type="email" name="email" placeholder="Your email" required>
#         <textarea name="message" placeholder="Your message here" required></textarea>
#         <button type="submit">Send</button>
#     </form>
#     """
#     left_column, right_column = st.columns(2)
#     with left_column:
#         st.markdown(contact_form, unsafe_allow_html=True)
#     with right_column:
#         st.empty()


