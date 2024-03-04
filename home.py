import streamlit as st

def show():
    st.markdown("""
        <h1>Welcome to <span style='color: #1ed760'>SENTUNE</span>! 🎶</h1>
        <div>Use the sidebar according to your requirements:</div>
        <ul>
            <li>Choose: find a playlist that identifies with your desired audience.</li>
            <li>Analyse: obtain insights from a particular playlist.</li>
            <li>Single Song: obtain insights from a particular song.</li>
            <li>User Manual: read our manual if you require help in understanding or using the app interface.</li>
        </ul>
        """, unsafe_allow_html=True)
    
    # Add this line at the bottom to display the emotions.webp image
    st.image('emotions.webp', caption='6 Types of Basic Emotions')
