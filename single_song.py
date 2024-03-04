import streamlit as st
import matplotlib.pyplot as plt
from utils import (df, process_text_input, chunk_text, average_sentiment, song_sentiment_pie_chart, word_importance_scores, 
                    all_sent_importance_viz, sent_specific_importance_viz, set_specific_importance_histogram)
from transformers import pipeline, AutoTokenizer
import numpy as np
import pandas as pd
import string
from IPython.display import HTML
import matplotlib.pyplot as plt
import plotly.graph_objects as go

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

"""# Reusable Variables"""

# Define sentiment-to-color mapping
sentiment_colors = {
    'joy': '#FE6C40',
    'sadness': '#7CAEFF',
    'fear': '#FFCF32',
    'disgust': '#ABD700',
    'anger': '#A102FE',
    'surprise': '#FF86C3',
    'neutral': '#D3D0D0'
}

max_chunk_size = 256
overlap_size = 50


#test chat gpt:
# Initialize session state variables if they don't exist
if 'selected_song' not in st.session_state:
    st.session_state.selected_song = None
if 'lyrics' not in st.session_state:
    st.session_state.lyrics = None
if 'sentiment_scores' not in st.session_state:
    st.session_state.sentiment_scores = None
if 'importance_scores' not in st.session_state:
    st.session_state.importance_scores = None

def show():
    st.header("Analyze Single Song")

    # Dropdown for song selection
    song_selection = st.selectbox("Choose a song:", ["Select a song"] + sorted(df['track.name'].unique()))

    if song_selection != "Select a song":
        # Store the song selection in session state
        st.session_state.selected_song = song_selection
        # Assuming 'cleaned_lyrics' is a column in your dataframe
        lyrics = df[df['track.name'] == st.session_state.selected_song].iloc[0]['cleaned_lyrics']
        # Truncate the lyrics to a maximum of 775 characters
        truncated_lyrics = lyrics[:775] if len(lyrics) > 775 else lyrics
        st.session_state.lyrics = truncated_lyrics
        # st.session_state.lyrics = lyrics
        st.write("Lyrics:")
        #st.text(st.session_state.lyrics)
        # Use st.text_area to display the lyrics, set the height appropriately
        st.text_area("", st.session_state.lyrics, height=300, key="lyrics_display", disabled=True)

    if st.button("Analyze Emotions"):
        # Perform and store analysis results in session state
        '''
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")   
        max_chunk_size = 256
        overlap_size = 50
        '''
        lyrics_processed = process_text_input(st.session_state.lyrics)
        st.session_state.sentiment_scores = average_sentiment(lyrics_processed)
        st.session_state.importance_scores = word_importance_scores(lyrics_processed,classifier, tokenizer)

        # Display Pie Chart
        pie_chart = song_sentiment_pie_chart(st.session_state.sentiment_scores, sentiment_colors)
        st.plotly_chart(pie_chart)

        # Display Highlighted Lyrics (All Sentiments)
        highlighted_lyrics_all, legend = all_sent_importance_viz(sentiment_colors, st.session_state.lyrics, st.session_state.importance_scores)
        st.markdown(highlighted_lyrics_all, unsafe_allow_html=True)
        st.markdown(legend, unsafe_allow_html=True)

    # Dropdown for detailed sentiment analysis
    if 'sentiment_scores' in st.session_state and st.session_state.sentiment_scores is not None:
        selected_sentiment = st.selectbox("Select a sentiment for detailed analysis:", ["Select a sentiment"] + list(sentiment_colors.keys()))
        if selected_sentiment != "Select a sentiment":
            # Display sentiment-specific visualization and histogram using stored results
            highlighted_lyrics_specific, legend_html_content = sent_specific_importance_viz(st.session_state.lyrics, selected_sentiment, st.session_state.importance_scores, sentiment_colors)
            st.markdown(highlighted_lyrics_specific, unsafe_allow_html=True)
            st.markdown(legend_html_content, unsafe_allow_html=True)



            #highlighted_lyrics_specific = sent_specific_importance_viz(st.session_state.lyrics, selected_sentiment, st.session_state.importance_scores, sentiment_colors)
            #st.markdown(highlighted_lyrics_specific, unsafe_allow_html=True)
            histogram_fig = set_specific_importance_histogram(st.session_state.importance_scores, selected_sentiment, sentiment_colors)
            st.plotly_chart(histogram_fig)
            

show()









'''
"""# Execution Codes"""

# Process Manual Input of Multi-line Lyrics to Single-Line String Variable 'lyrics'

lyrics = process_text_input(lyrics)

# Obtain Sentiment Analysis Scores for 'lyrics'
sentiment_scores = average_sentiment(lyrics)

# Display Song Sentiment Analysis Pie Chart
song_sentiment_pie_chart(sentiment_scores, sentiment_colors)

# Calculate the importance scores for each word within 'lyrics'
importance_scores = word_importance_scores(lyrics, classifier, tokenizer)

# Execute the function
all_sent_importance_viz(sentiment_colors, lyrics, importance_scores)

# Choose the sentiment for the Sentiment-Specific Word Importance Visualization
chosen_sentiment = 'sadness'  # Change this to the sentiment you want to highlight

# Display Sentiment-Specific Word Importance Visualization
sent_specific_importance_viz(lyrics, chosen_sentiment, importance_scores, sentiment_colors)

# Display set_specific_importance_histogram
set_specific_importance_histogram(importance_scores, chosen_sentiment, sentiment_colors)
'''

