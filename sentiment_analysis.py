import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import plotly.graph_objects as go
from transformers import pipeline, AutoTokenizer
import numpy as np
import string
import torch
import zipfile
import os

#Loading of model and data
def load_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    classifier = pipeline("text-classification", model=model_name, return_all_scores=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return classifier, tokenizer

classifier, tokenizer = load_model()

# Function to load data
def load_data():
    # Path to the compressed folder (.zip file)
    compressed_folder_path = 'final_lyrics_with_SA.csv.zip'

    # Extract the compressed folder
    with zipfile.ZipFile(compressed_folder_path, 'r') as zip_ref:
        # Assuming there's only one file in the compressed folder
        compressed_file = zip_ref.namelist()[0]
        zip_ref.extract(compressed_file)

        # Read the CSV file
        data = pd.read_csv(compressed_file)

    # Remove the extracted file after reading
    os.remove(compressed_file)

    return data

# Load data
df = load_data()

# Preprocess the lyrics column to remove '#' symbols
df['cleaned_lyrics'] = df['consolidated_lyrics'].str.replace('#', '')

# Mapping sentiments to their respective colors
sentiment_colors = {
    'anger': '#A102FE',
    'disgust': '#ABD700',
    'fear': '#FFCF32',
    'joy': '#FE6C40',
    'neutral': '#D3D0D0',
    'sadness': '#7CAEFF',
    'surprise': '#FF86C3'
}

if 'selected_sentiment' not in st.session_state:
    st.session_state['selected_sentiment'] = ''

def extract_keywords(lyrics_series):
    # Combine all lyrics into one text
    text = ' '.join(lyrics_series.dropna())
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Split the text into words and convert to lowercase
    words = text.lower().split()
    # Count the occurrences of each word
    word_counts = Counter(words)
    # Return the most common words
    return word_counts.most_common(10)

def chunk_text(text, max_chunk_size, overlap_size, tokenizer):
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)
    
    # Adjust max_chunk_size to account for special tokens ([CLS], [SEP], <s>, </s>, etc.)
    # This example assumes 2 special tokens at the start and end. Adjust if your model uses a different format.
    max_chunk_size -= 2

    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        # Find the end index of the chunk, ensuring we do not exceed the token list's length
        end_idx = start_idx + max_chunk_size
        if end_idx > len(tokens):
            end_idx = len(tokens)

        # Prepare the chunk with the appropriate special tokens
        chunk = [tokenizer.cls_token] + tokens[start_idx:end_idx] + [tokenizer.sep_token]
        
        # Convert the chunk back to a string format suitable for the model
        chunk_str = tokenizer.convert_tokens_to_string(chunk)
        
        # Append the chunk string to the list of chunks
        chunks.append(chunk_str)

        # Update the start index for the next chunk, using overlap if specified
        start_idx = end_idx - overlap_size

    return chunks

def average_sentiment(text, tokenizer, classifier, max_chunk_size=512, overlap_size=50):
    # Split the text into chunks based on the max_chunk_size and overlap_size
    chunks = chunk_text(text, max_chunk_size, overlap_size, tokenizer)
    total_scores = [0] * 7  # Initialize total scores for each sentiment category
    num_chunks = len(chunks)

    for chunk in chunks:
        # Calculate sentiment scores for each chunk
        try:
            scores = classifier(chunk)
            for i, score in enumerate(scores[0]):
                total_scores[i] += score['score']
        except Exception as e:
            st.error(f"An error occurred during sentiment analysis for chunk: {chunk}")
            st.error(e)

    average_scores = [score / num_chunks for score in total_scores]
    return average_scores

#Functions that will be called from different sections
def map_token_scores_to_words(tokens, token_scores, original_words):
    word_scores = {}
    # Assuming that the tokenizer does not split words and that tokens and original_words have a 1-1 mapping
    for i, word in enumerate(original_words):
        word_scores[word] = token_scores[i]  # Directly map the score of the token to the word
    return word_scores

def sent_specific_importance_viz(original_words, word_scores, sentiment_colors, importance_scores, selected_sentiment):
    highlighted_lyrics = "<div style='white-space: pre-wrap;'>"  # Use white-space: pre-wrap to preserve line breaks
    print("Selected sentiment:", selected_sentiment)
    print("Sentiment colors:", sentiment_colors)
    print("Word scores:", word_scores)

    for word in original_words:
        score = word_scores.get(word, 0.5)  # Use the score from the mapped word_scores
        sentiment = selected_sentiment.lower()
        color = sentiment_colors.get(sentiment, '#FFFFFF')  # Default to white color if sentiment is not found
        opacity = (score - 0.5) * 2  # Adjust opacity to reflect the score (assuming score is normalized)
        background_color = f"rgba({color[1:]},{opacity:.2f})"
        highlighted_lyrics += f"<span style='background-color:{background_color};'>{word} </span>"
    # Remove consecutive spaces and replace them with a single space
    highlighted_lyrics = re.sub(r'\s+', ' ', highlighted_lyrics)
    highlighted_lyrics += "</div>"
    return highlighted_lyrics

def normalize_scores(scores):
    if isinstance(scores, dict):
        max_score = max(scores.values()) if scores else 0
        min_score = min(scores.values()) if scores else 0
        return {word: (score - min_score) / (max_score - min_score) if max_score > min_score else 0.5 for word, score in scores.items()}
    else:
        return scores

def create_and_display_pie_chart(sentiments, sentiment_values, colors):
    fig = go.Figure(data=[go.Pie(labels=sentiments, values=sentiment_values, marker=dict(colors=list(colors)))])
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig)

# Function to process text input
def process_text_input(multiline_text):
    # Split the multiline text into lines
    lines = multiline_text.split('\n')
    # Join the lines into a single string with spaces between them
    single_line_text = ' '.join(lines)
    return single_line_text

# Function to perform sentiment analysis and visualization
def perform_sentiment_analysis(lyrics, tokenizer, classifier):
    # Calculate sentiment scores
    sentiment_scores = average_sentiment(lyrics, tokenizer, classifier)
    
    # Pie chart visualization
    labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Neutral', 'Sadness', 'Surprise']
    sentiment_percentages = [(score / sum(sentiment_scores)) * 100 for score in sentiment_scores]
    colors = ['#A102FE', '#ABD700', '#FFCF32', '#FE6C40', '#D3D0D0', '#7CAEFF', '#FF86C3']
    fig_pie_chart = go.Figure(data=[go.Pie(labels=labels, values=sentiment_percentages, marker=dict(colors=colors))])
            
    # Word importance scores
    importance_scores = word_importance_scores(lyrics)
    
    # Word-by-word visualization
    html_content = "<div style='font-size:16px;'>"
    for word in lyrics.split():
        clean_word = word.strip(',.?!\"')
        normalized_score = normalized_scores.get(clean_word, 0)
        opacity = int(min_opacity + (normalized_score * (max_opacity - min_opacity)))
        opacity = max(min(opacity, 255), 0)
        highlight_color_with_opacity = f"{sentiment_colors[chosen_sentiment]}{opacity:02X}"
        background_style = f"background-color:{highlight_color_with_opacity};"
        html_content += f"<span style='{background_style}'>{word} </span>"
    html_content += "</div>"
    
    # Distribution of sentiment values - Matplotlib
    sentiment_values = [scores[chosen_sentiment] for scores in importance_scores.values()]
    plt.figure(figsize=(10, 6))
    plt.hist(sentiment_values, bins=20, color=sentiment_colors[chosen_sentiment], edgecolor='black')
    plt.title(f'Distribution of Sentiment Values for {chosen_sentiment.capitalize()} Sentiment')
    plt.xlabel('Sentiment Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.xticks(fontsize=10, rotation=45)
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.6f'))
    fig_matplotlib = plt

    # Distribution of sentiment values - Plotly
    fig_plotly = go.Figure(data=[go.Histogram(x=sentiment_values, nbinsx=20, marker_color=sentiment_colors[chosen_sentiment])])

    # Word by word impact on overall sentiment scoring
    legend_html_content = "<div style='font-size:14px; margin-top:10px;'>"
    sorted_sentiments = sorted(max_scores.keys(), key=lambda x: max_scores[x], reverse=True)
    for index, sentiment in enumerate(sorted_sentiments, start=1):
        max_score = max_scores[sentiment]
        min_score = min_scores[sentiment]
        legend_html_content += f"<b>{index}. {sentiment.capitalize()}</b><br>"
        max_highlight_color_with_opacity = f"{sentiment_colors[sentiment]}{max_opacity:02X}"
        max_background_style = f"color:black; background-color:{max_highlight_color_with_opacity}; opacity: {max_opacity:.2f}"
        legend_html_content += f"<span style='{max_background_style}'>Maximum value: {max_score:.6f}</span><br>"
        min_highlight_color_with_opacity = f"{sentiment_colors[sentiment]}{min_opacity:02X}"
        min_background_style = f"color:black; background-color:{min_highlight_color_with_opacity}; opacity: {min_opacity:.2f}"
        legend_html_content += f"<span style='{min_background_style}'>Minimum value: {min_score:.6f}</span><br><br>"
    legend_html_content += "</div>"

    return fig_pie_chart, html_content, fig_matplotlib, fig_plotly, legend_html_content

def word_importance_scores(text, classifier, tokenizer):
    # Remove '#' symbols from the text
    text = text.replace('#', '')
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Initialize the dictionary to store importance scores
    importance_scores = {}

    # Process text in chunks to avoid exceeding the model's maximum input length
    for i in range(0, len(tokens), tokenizer.model_max_length):
        # Select a chunk of tokens
        chunk_tokens = tokens[i:i + tokenizer.model_max_length]

        # Get baseline sentiment scores for the original chunk
        original_scores = classifier(chunk_tokens, return_all_scores=True)[0]

        # Iterate over each token in the chunk to measure its importance
        for j, token in enumerate(chunk_tokens):
            # Exclude special tokens from importance score calculation
            if token not in tokenizer.special_tokens_map.values():
                # Create a perturbed version of the chunk by replacing the current token with the [MASK] token
                perturbed_chunk = chunk_tokens[:j] + [tokenizer.mask_token] + chunk_tokens[j + 1:]
                perturbed_text = tokenizer.convert_tokens_to_string(perturbed_chunk)

                # Get sentiment scores for the perturbed chunk
                perturbed_scores = classifier(perturbed_text, return_all_scores=True)[0]

                # Calculate the difference in scores caused by perturbing the token
                score_diff = {score['label']: abs(original_score['score'] - score['score'])
                              for original_score, score in zip(original_scores, perturbed_scores)}

                # Update the importance scores, using the original token
                importance_scores[token] = score_diff

    return importance_scores

##################################################################Start from Streamlit sections
def app_layout():
    # Sidebar for navigation
    app_mode = st.sidebar.radio("Navigate", ["Choose", "Analyze", "Single Song", "User Manual"])

    # Pages content
    if app_mode == "Choose":
        choose_page()
    elif app_mode == "Analyze":
        analyze_page()
    elif app_mode == "Single Song":
        single_song_page()
    elif app_mode == "User Manual":
        user_manual_page()


def choose_page():
    st.header("Playlist Finder")
    st.write('Filters')
    col1, col2 = st.columns(2)

    with col1:
        music_type = st.selectbox("By Music Type:", options=['All'] + sorted(df['track_genre'].unique().tolist()))

    with col2:
        keyword = st.text_input('Keyword (in playlist name)')

    st.title('')

    st.write('')

    if st.button('Get Results'):
        filtered_df = df  # Initialize filtered_df with df to handle the case where no filters are applied yet
        if music_type != 'All':
            filtered_df = filtered_df[filtered_df['track_genre'].str.contains(music_type, case=False, na=False, regex=False)]
        if keyword:
            filtered_df = filtered_df[filtered_df['playlist.name'].str.contains(keyword, case=False, na=False, regex=False)]

        # Group by playlist.name and aggregate details
        grouped_df = filtered_df.groupby('playlist.name').agg({
            'track_genre': lambda x: ', '.join(x.unique()),
            'artist_name': lambda x: ', '.join(x.unique()),
            'track.name': lambda x: ', '.join(x.unique())  # Assuming you have a column 'track.name' for song titles
        }).reset_index()

        if not grouped_df.empty:
            display_df = grouped_df.rename(columns={
                'playlist.name': 'Playlist Name',
                'track_genre': 'Genre',
                'country': 'Country',
                'artist_name': 'Artists Name',
                'track.name': 'Songs'  # Assuming 'track.name' holds the song names
            })

            # Display the DataFrame with specified width and height
            st.table(display_df, )

            # Extract playlist names for the multiselect widget, using the new column name
            playlist_names = display_df['Playlist Name'].tolist()
            selected_playlists = st.multiselect('Select Playlists', options=playlist_names)

            # Display selected playlists details
            if selected_playlists:
                st.write("Selected Playlists Details")
                songs = []
                for playlist in selected_playlists:
                    playlist_details = display_df[display_df['Playlist Name'] == playlist]
                    songs.extend(playlist_details['Songs'].tolist())  # Assuming 'Songs' holds individual song names
                # Store the unique songs in session state
                st.session_state['selected_songs'] = list(set(songs))
        else:
            st.write('No playlists match your criteria.')

    st.write('---')

def analyze_page():
    st.header("Analyze Playlists")

    # Sidebar for selecting playlists
    selected_playlist = st.sidebar.selectbox("Select Playlist:", df['playlist.name'].unique())

    # Check if a playlist has been selected
    if not selected_playlist:
        st.warning("No playlist selected. Please choose a playlist from the sidebar.")
        return
  
    # Filter dataframe based on selected playlist
    playlist_df = df[df['playlist.name'] == selected_playlist]

    # Use two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Artists")
        artist_counts = Counter(playlist_df['artist_name'])
        top_artists = artist_counts.most_common(5)  # Limiting to top 5 results
        top_artists_df = pd.DataFrame(top_artists, columns=['Artist', 'Frequency'])
        st.table(top_artists_df)
        st.write('---')

    with col2:
        st.subheader("Top Genres")
        genre_counts = Counter(playlist_df['track_genre'])
        top_genres = genre_counts.most_common(5)  # Limiting to top 5 results
        top_genres_df = pd.DataFrame(top_genres, columns=['Genre', 'Frequency'])
        st.table(top_genres_df)
        st.write('---')

    with col1:
        st.subheader("Speech - Keywords")
        # Extract keywords from the lyrics, now omitting stopwords
        keywords = extract_keywords(playlist_df['consolidated_lyrics'])
        # Create a bar chart for the top N keywords
        fig, ax = plt.subplots()
        words, counts = zip(*keywords)
        ax.bar(words, counts)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)  

    with col2:
        #st.subheader("Danceability Score")
        average_danceability = playlist_df['danceability'].mean()
        st.metric(label="Average Danceability", value=f"{average_danceability:.2f}")

        st.progress(average_danceability, text=None)

    with col2:
        #st.subheader("Energy Score")
        average_energy = playlist_df['energy'].mean()
        st.metric(label="Average Energy", value=f"{average_energy:.2f}")

        st.progress(average_energy, text=None)

    with col2:
        #st.subheader("Acousticness")
        average_acousticness = playlist_df['acousticness'].mean()
        st.metric(label="Average Acousticness", value=f"{average_acousticness:.2f}")
        st.progress(average_acousticness)

    with col2:
        #st.subheader("Instrumentalness")
        average_instrumentalness = playlist_df['instrumentalness'].mean()
        st.metric(label="Average Instrumentalness", value=f"{average_instrumentalness:.2f}")
        st.progress(average_instrumentalness)

    with col2:
        #st.subheader("Liveness")
        average_liveness = playlist_df['liveness'].mean()
        # Inject the style dynamically 
        st.metric(label="Average Liveness", value=f"{average_liveness:.2f}")
        st.progress(average_liveness)

    with col2:
        #st.subheader("Tempo")
        average_tempo = playlist_df['tempo'].mean()
        st.metric(label="Average Tempo (BPM)", value=f"{average_tempo:.0f}")

    with col2:
        #st.subheader("Valence")
        average_valence = playlist_df['valence'].mean()
        st.metric(label="Average Valence", value=f"{average_valence:.2f}")
        st.progress(average_valence)

def single_song_page():
    st.header("Analyze Single Song")
    song = None  # Initialize song variable
    lyrics = ""  # Initialize lyrics variable for manual input case

    if 'selected_songs' in st.session_state and st.session_state['selected_songs']:
        song = st.selectbox("Choose a song from your selected playlists:", st.session_state['selected_songs'])
    else:
        song_selection_method = st.radio("Select a song by:", ("Choosing from list", "Inputting manually"))
        if song_selection_method == "Choosing from list":
            song = st.selectbox("Choose a song:", df['track.name'].unique())
            if song:
                lyrics = df[df['track.name'] == song].iloc[0]['cleaned_lyrics']  # Use cleaned_lyrics column for chosen song
        elif song_selection_method == "Inputting manually":
            lyrics = st.text_area("Paste the lyrics of the song you want to analyze:", height=150)
            if st.button("Analyze Lyrics", key="analyze_manually"):
                st.session_state['analyzed_sentiments'] = word_importance_scores(lyrics, classifier, tokenizer)
                st.session_state['analyzed_song'] = "manual_input"
        else:
            song_name_input = st.text_input("Input song name:")
            if st.button("Search"):
                song_data = df[df['track.name'].str.lower() == song_name_input.lower()]
                if not song_data.empty:
                    song = song_data.iloc[0]['track.name']
                    lyrics = song_data.iloc[0]['cleaned_lyrics']
                else:
                    st.write("Song not found.")
                    return

    if lyrics:
        st.subheader("Lyrics")
        st.write(lyrics)
        if song or song_selection_method == "Inputting manually":
            if 'analyzed_sentiments' not in st.session_state or st.session_state['analyzed_song'] != song:
                analyze_emotions = st.button("Analyze Emotions")
                if analyze_emotions or song_selection_method == "Inputting manually":
                    st.session_state['analyzed_sentiments'] = word_importance_scores(lyrics, classifier, tokenizer)
                    st.session_state['analyzed_song'] = song if song else "manual_input"

    if 'analyzed_sentiments' in st.session_state and (st.session_state.get('analyzed_song') == song or st.session_state.get('analyzed_song') == "manual_input"):
        # Generate and display the pie chart
        sentiment_values = [sum(st.session_state['analyzed_sentiments'][word].get(sentiment, 0) for word in st.session_state['analyzed_sentiments']) / len(lyrics.split()) for sentiment in sentiment_colors.keys()]
        pie_chart = create_and_display_pie_chart(list(sentiment_colors.keys()), sentiment_values, list(sentiment_colors.values()))
        if pie_chart:
            st.plotly_chart(pie_chart)

        st.markdown("Select a sentiment to analyze:")
        selected_sentiment = st.selectbox("Choose a sentiment to analyze:", list(sentiment_colors.keys()), key='sentiment_selection', on_change=None)

        if selected_sentiment:
            selected_sentiment_scores = {word: scores.get(selected_sentiment.lower(), 0) for word, scores in st.session_state['analyzed_sentiments'].items() if scores.get(selected_sentiment.lower(), 0) != 0}

            if selected_sentiment_scores:
                scores_values = list(selected_sentiment_scores.values())
                plt.figure()
                plt.hist(scores_values, bins=20, color=sentiment_colors[selected_sentiment.lower()], edgecolor='black')
                plt.title(f'Distribution of Sentiment Scores for {selected_sentiment.capitalize()} Sentiment Words')
                plt.xlabel('Sentiment Score')
                plt.ylabel('Frequency')
                st.pyplot(plt)
            else:
                st.write(f"No words found for the {selected_sentiment.capitalize()} sentiment in the lyrics.")

def user_manual_page():
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

# Main function to run the app
if __name__ == "__main__":
    st.sidebar.title("Playlist Insights")
    app_layout()
