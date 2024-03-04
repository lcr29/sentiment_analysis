
import streamlit as st

'''
def show():
    st.title("User Manual")

    st.header("Introduction")
    st.write("This is the user manual for our Streamlit app. Use the sidebar to navigate through different pages.")

    st.header("Navigating the App")
    st.write("Description of how to navigate the app and use its features.")

    st.header("FAQ")
    st.write("Answers to frequently asked questions.")

    st.header("Contact")
    st.write("Contact information for further support.")

'''


def show():
    st.title("SENTUNE ANALYTICS")
    st.title("📘 User Manual")

    st.header("🌟 Introduction")
    st.write("""
        Welcome to SENTUNE ANALYTICS! This manual guides you through using the SENTUNE application 
        to analyze song playlists and lyrics. Whether you're interested in understanding the mood of music 
        for marketing, political messaging, or just out of curiosity, SENTUNE makes it easy and insightful. 🎶
    """)

    st.header("🔍 Searching for Playlists")
    st.write("""
        - **Select Filters**: Use the filter options to narrow down your search. You can filter by:
          - Music Type: Choose genres like 'alt-rock', 'alternative', etc.
          - Country: Select playlists based on the country.
          - Keyword: Look for specific keywords in playlist names.
        - **Get Results**: After setting your filters, click the 'Get Results' button to view matching 
          playlists.
    """)

    st.header("📊 Playlist Insights")
    st.write("""
        - **Choose a Playlist**: Pick a playlist from the ones you've found or select from existing ones.
        - **Analyze Playlists**: Analyze playlists for keywords and overall sentiment. For example, see 
          what the 'Top 50 - Belgium' playlist reveals about the lyrics' mood.
        - **Compare Playlists**: Compare different playlists to see how they differ in mood and 
          keywords.
        - **Single Song Analysis**: Select a single song for a detailed breakdown of its sentiments.
    """)

    st.header("📈 Graphical Analysis")
    st.write("""
        - **Speech - Keywords**: View bar graphs showing the most important keywords in songs.
        - **Speech - Sentiment**: Understand the emotional tone of lyrics through sentiment graphs.
        - **Sentiment Distribution for Single Song**: See a pie chart that breaks down various 
          sentiments in a song's lyrics.
    """)

    st.header("👩‍💻 Navigating the User Interface")
    st.write("""
        - **Sidebar**: Use the sidebar to switch between 'Choose', 'Analyze', 'Compare', 'Single Song', 
          and 'User Manual' modes.
        - **Dropdown Menus**: Select playlists and songs from dropdown menus.
        - **Execution Button**: Use the 'Get Results' button to execute searches or analyses.
        - **Layout**: The application is designed to be clear and intuitive, guiding you through each step.
    """)

    st.header("💡 Tips for Best Experience")
    st.write("""
        - **Explore Different Filters**: Don't hesitate to try various combinations of filters for diverse 
          insights.
        - **Regular Updates**: Check for app updates regularly for the latest features and improved 
          experiences.
        - **Feedback**: Your feedback is valuable. If you encounter issues or have suggestions, please 
          reach out through the app's feedback section.
    """)

    st.header("🎉 Conclusion")
    st.write("""
        SENTUNE ANALYTICS is your gateway to understanding the emotional landscape of music. 
        Whether you're tailoring a marketing campaign or exploring musical sentiments for personal 
        interest, SENTUNE provides a unique and powerful tool. Enjoy your journey into the world of music and emotions.
    """)