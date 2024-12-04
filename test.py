import time
import streamlit as st

# Function to clear the input field
def clear_search():
    # st.session_state.search_query = ""  # Clear the input field
    del st.session_state.search_query

# Add a text input with a placeholder and a default empty value
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

input_string = st.text_input(
    "Search",
    placeholder="🔍 Type your query here...",
    key="search_query"
)

# Process the input if it's not empty
if st.session_state.search_query:
    st.write(f"Processing your query: {st.session_state.search_query}")
    st.success("Query processed successfully!")
    time.sleep(2)  # Simulate some processing delay
    clear_search()  # Clear the input after processing
    st.rerun()
