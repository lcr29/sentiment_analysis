import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Import Plotly graph objects
from collections import Counter
from utils import df, sentiment_colors  # Ensure utils.py has the required imports and data

def show():
    st.header("Analyze Playlists")

    selected_playlist = st.sidebar.selectbox("Select Playlist:", df['playlist.name'].unique())
    keywords_df = pd.read_csv('playlist_top_keywords_with_freq.csv')

    if not selected_playlist:
        st.warning("No playlist selected. Please choose a playlist from the sidebar.")
        return

    # Display the selected playlist as a title for information
    st.subheader(f"You are currently nalyzing the playlist: {selected_playlist}")
    st.write("Want to see more? Select another playlist in the dropdown menu on your left if you want!.")

    playlist_df = df[df['playlist.name'] == selected_playlist]

    # Filter keywords_df for the selected playlist
    playlist_keywords_df = keywords_df[keywords_df['playlist.name'] == selected_playlist]

    # Calculate average sentiment
    sentiment_cols = ['joy', 'fear', 'sadness', 'anger', 'surprise', 'neutral', 'disgust']
    sentiment_df = playlist_df[sentiment_cols].mean().reset_index()
    sentiment_df.columns = ['sentiment', 'value']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Speech - Keywords")
        if not playlist_keywords_df.empty:
            # Extract top 10 keywords and their frequencies for the selected playlist
            keywords = []
            frequencies = []
            for i in range(1, 11):  # Loop through top 10 keywords
                keyword_column = f'TOP_{i}_keyword'
                frequency_column = f'TOP_{i}_keyword_freq'
                if keyword_column in playlist_keywords_df.columns and frequency_column in playlist_keywords_df.columns:
                    keyword = playlist_keywords_df[keyword_column].iloc[0]  # Use .iloc[0] to get the first row's value
                    frequency = playlist_keywords_df[frequency_column].iloc[0]
                    keywords.append(keyword)
                    frequencies.append(frequency)
            
            # Create and display the histogram using Plotly
            fig_keywords = go.Figure(data=[go.Bar(x=keywords, y=frequencies)])
            fig_keywords.update_layout(title_text='Top Keywords Frequency', xaxis_title="Keywords", yaxis_title="Frequency", width=350, height=350)
            st.plotly_chart(fig_keywords)
        else:
            st.write("No keyword data available for this playlist.")

    with col2:
       st.subheader("Sentiment Analysis")
       fig_sentiment = px.pie(sentiment_df, names='sentiment', values='value', title='Average Sentiment Distribution', color='sentiment',
                               color_discrete_map=sentiment_colors)
       fig_sentiment.update_layout(width=350, height=350)  # Adjust size to fit column
       st.plotly_chart(fig_sentiment)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Audio Features Metrics")
        metrics = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
        metrics_df = playlist_df[metrics].mean().reset_index()
        metrics_df.columns = ['feature', 'value']
        fig_metrics = px.bar(metrics_df, x='value', y='feature', orientation='h', title="Audio Features", range_x=[0,1])
        fig_metrics.update_layout(width=350, height=350)  # Adjust size to fit column
        st.plotly_chart(fig_metrics)

        # Display Tempo and Loudness separately
        st.metric("Tempo", f"{playlist_df['tempo'].mean():.2f} BPM")
        st.metric("Loudness", f"{playlist_df['loudness'].mean():.2f} dB")

    with col4:
        st.subheader("Top Artists and Genres")
        artist_counts = Counter(playlist_df['artist_name'])
        top_artists = artist_counts.most_common(5)
        top_artists_df = pd.DataFrame(top_artists, columns=['Artist', 'Frequency'])
        genre_counts = Counter(playlist_df['track_genre'])
        top_genres = genre_counts.most_common(5)
        top_genres_df = pd.DataFrame(top_genres, columns=['Genre', 'Frequency'])
        st.write("Top Artists:")
        st.dataframe(top_artists_df)
        st.write("Top Genres:")
        st.dataframe(top_genres_df)

show()
















'''

 with col1:
        st.subheader("Speech - Keywords")
        keywords = []
        frequencies = []
        for i in range(1, 11):  # Loop through top 10 keywords
            keyword = keywords_df[f'TOP_{i}_keyword'].values[0]
            frequency = keywords_df[f'TOP_{i}_keyword_freq'].values[0]
            keywords.append(keyword)
            frequencies.append(frequency)


def show():
    st.header("Analyze Playlists")

    # Assuming 'playlist_top_keywords_with_freq.csv' is in the same directory
    keywords_df = pd.read_csv('playlist_top_keywords_with_freq.csv')

    selected_playlist = st.sidebar.selectbox("Select Playlist:", df['playlist.name'].unique())

    if not selected_playlist:
        st.warning("No playlist selected. Please choose a playlist from the sidebar.")
        return

    playlist_df = df[df['playlist.name'] == selected_playlist]

    # Assuming keywords_df has a 'playlist.name' column
    playlist_keywords_df = keywords_df[keywords_df['playlist.name'] == selected_playlist]

    if not playlist_keywords_df.empty:
        keywords = []
        frequencies = []
        for i in range(1, 11):  # Loop through top 10 keywords
            keyword = playlist_keywords_df[f'TOP_{i}_keyword'].values[0]
            frequency = playlist_keywords_df[f'TOP_{i}_keyword_freq'].values[0]
            keywords.append(keyword)
            frequencies.append(frequency)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Speech - Keywords")
            fig, ax = plt.subplots()
            ax.bar(keywords, frequencies)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

    with col1:
        st.subheader("Top Artists")
        artist_counts = Counter(playlist_df['artist_name'])
        top_artists = artist_counts.most_common(5)
        top_artists_df = pd.DataFrame(top_artists, columns=['Artist', 'Frequency'])
        st.table(top_artists_df)

    with col2:
        st.subheader("Top Genres")
        genre_counts = Counter(playlist_df['track_genre'])
        top_genres = genre_counts.most_common(5)
        top_genres_df = pd.DataFrame(top_genres, columns=['Genre', 'Frequency'])
        st.table(top_genres_df)

    # Additional metrics based on your data columns
    # Replace 'danceability', 'energy', etc. with actual columns in your dataset
    with col2:
        metrics = [("Danceability", 'danceability'), ("Energy", 'energy'), ("Acousticness", 'acousticness'),
                   ("Instrumentalness", 'instrumentalness'), ("Liveness", 'liveness'), ("Tempo", 'tempo'), ("Valence", 'valence')]
        for label, column in metrics:
            if column in playlist_df.columns:
                average_value = playlist_df[column].mean()
                st.metric(label=f"Average {label}", value=f"{average_value:.2f}")
                # Assuming you want to display a progress bar for each metric
                st.progress(min(int(average_value * 100), 100))

'''
