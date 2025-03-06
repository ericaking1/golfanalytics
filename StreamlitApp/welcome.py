import streamlit as st

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.header('Golf Analytics Dashboard')
st.write("Welcome!")
st.text('This dashbaord is contructed by Amy Wang and Erica King, utilizing research from our golf analytics project in the Johns Hopkins University Sports Analytics Research Group led by Dr. Anton Dahbura. We use machine learning and artificial intelligence to analyze the anatomies of the dwon swing and aim to provide feedback on how to best improve. \n')
st.text('To best utilize this dashboard, please upload a csv file of your shot history for the features that we analyze. \n')