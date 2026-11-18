import streamlit as st

st.set_page_config(page_title="Streamlit Test")

st.title("Streamlit is Working!")
st.success("Application rendered successfully.")

st.write("Testing our Supply Chain Control Tower.")

name = st.text_input("Enter your name")

if name:
    st.write(f"Welcome, {name}!")

st.button("Test Button")
