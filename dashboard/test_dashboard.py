import streamlit as st

st.set_page_config(layout="wide")

st.title("Supply Chain Dashboard Test")

st.success("Dashboard folder is working!")

st.write("Testing Python execution inside the dashboard directory.")

st.metric("Test Orders", "96,470")

st.bar_chart({
    "Orders": [100, 200, 300, 400, 500]
})