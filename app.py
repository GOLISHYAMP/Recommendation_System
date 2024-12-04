import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import os
from io import BytesIO

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd

base_path = os.getcwd()
sub_path = 'data'
dataset_path = os.path.join(base_path, sub_path, 'mobile_recommendation_system_dataset.csv')

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

lottie_coding = load_lottiefile(os.path.join(os.getcwd(), "Animation - 1733047553544.json"))


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- LOAD ASSETS ----
# lottie_coding = load_lottieurl("https://lottie.host/b5066c54-82de-4c32-93e0-046f9c4a17c9/gUv6XtilSp.lottie")
# img_contact_form = Image.open("images/yt_contact_form.png")
# img_lottie_animation = Image.open("images/yt_lottie_animation.png")
img_contact_form = load_lottiefile(os.path.join(os.getcwd(), "Animation - 1733047553544.json"))
img_lottie_animation = load_lottiefile(os.path.join(os.getcwd(), "Animation - 1733047553544.json"))
# ---- HEADER SECTION ----
with st.container():
    st.subheader("Hi, I am Sven :wave:")
    st.title("A Data Analyst From Germany")
    st.write(
        "I am passionate about finding ways to use Python and VBA to be more efficient and effective in business settings."
    )
    st.write("[Learn More >](https://pythonandvba.com)")
 
# ---- WHAT this project does explaination ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What I do")
        st.write("##")
        st.write(
            """
            On my YouTube channel I am creating tutorials for people who:
            - are looking for a way to leverage the power of Python in their day-to-day work.
            - are struggling with repetitive tasks in Excel and are looking for a way to use Python and VBA.
            - want to learn Data Analysis & Data Science to perform meaningful and impactful analyses.
            - are working with Excel and found themselves thinking - "there has to be a better way."

            If this sounds interesting to you, consider subscribing and turning on the notifications, so you don’t miss any content.
            """
        )
        st.write("[YouTube Channel >](https://youtube.com/c/CodingIsFun)")
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

# ---- Recommendations ----
import pickle
# Load the saved model
with open(r'src\model\similarity.pkl', 'rb') as file:
    similarity = pickle.load(file)

with open(r'src\model\name_model.pkl', 'rb') as file:
    name_model = pickle.load(file)

with open(r'src\model\name_vectors.pkl', 'rb') as file:
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
    print(card_text)
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


