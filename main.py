import streamlit as st
import home, choose, analyse, single_song, user_manual

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Choose", "Analyze", "Single Song", "User Manual"])

    if page == "Home":
        home.show()
    elif page == "Choose":
        choose.show()
    elif page == "Analyze":
        analyse.show()
    elif page == "Single Song":
        single_song.show()
    elif page == "User Manual":
        user_manual.show()

if __name__ == "__main__":
    main()
