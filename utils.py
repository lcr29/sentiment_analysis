import pandas as pd
import zipfile
import os
import re
from transformers import pipeline, AutoTokenizer
from collections import Counter
from transformers import pipeline, AutoTokenizer
import numpy as np
import pandas as pd
import string
from IPython.display import HTML
import matplotlib.pyplot as plt
import plotly.graph_objects as go

'''
# Loading of model and data
def load_model():
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    return classifier, tokenizer
'''
# classifier, tokenizer = load_model()

def load_data():
    compressed_folder_path = 'final_lyrics_with_SA.csv.zip'
    with zipfile.ZipFile(compressed_folder_path, 'r') as zip_ref:
        compressed_file = zip_ref.namelist()[0]
        zip_ref.extract(compressed_file)
        data = pd.read_csv(compressed_file)
    os.remove(compressed_file)
    return data

df = load_data()
df['cleaned_lyrics'] = df['consolidated_lyrics'].str.replace('#', '')

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

###

# Process Multiline Text Input to Single-Line String into 'lyrics' variable:

def process_text_input(multiline_text):
    # Split the multiline text into lines
    lines = multiline_text.split('\n')
    # Join the lines into a single string with spaces between them
    single_line_text = ' '.join(lines)
    return single_line_text

# Chunk Lyrics to Analyze Independently

def chunk_text(text, max_chunk_size = 256, overlap_size = 50):
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    tokens = tokenizer.tokenize(text)
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_chunk_size, len(tokens))
        chunks.append(tokens[start_idx:end_idx])
        start_idx += max_chunk_size - overlap_size
    return chunks

# Obtain Final Sentiment Scores for a Song's Lyrics (Average All Chunks)

def average_sentiment(text, max_chunk_size = 256, overlap_size = 50):
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    chunks = chunk_text(text, max_chunk_size, overlap_size)
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    total_scores = [0] * 7  # Initialize total scores for each sentiment category
    num_chunks = len(chunks)
    for chunk in chunks:
        chunk_text_str = tokenizer.convert_tokens_to_string(chunk)
        scores = classifier(chunk_text_str)
        for i, score in enumerate(scores[0]):
            total_scores[i] += score['score']
    average_scores = [score / num_chunks for score in total_scores]
    return average_scores

# Display Song Sentiment Pie Chart

def song_sentiment_pie_chart(sentiment_scores, sentiment_colors):
    total = sum(sentiment_scores)
    sentiment_percentages = [(score / total) * 100 for score in sentiment_scores]

    labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Neutral', 'Sadness', 'Surprise']

    colors = ['#A102FE', '#ABD700', '#FFCF32', '#FE6C40', '#D3D0D0', '#7CAEFF', '#FF86C3']

    fig = go.Figure(data=[go.Pie(labels=labels, values=sentiment_percentages, marker=dict(colors=colors))])
    return fig

# Obtain Word Importance Scores for a Song:

def word_importance_scores(text, classifier, tokenizer):
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")  
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    words = tokenizer.convert_tokens_to_string(tokens).split()

    # Remove punctuation from words
    words = [word.strip(string.punctuation) for word in words]

    # Get the original sentiment score for comparison
    original_scores = classifier(text)[0]

    # Initialize a dictionary to hold the importance scores for each word
    importance_scores = {}

    # Iterate through each word in the text
    for i, word in enumerate(words):
        if word in tokenizer.all_special_tokens:
            # Skip special tokens
            continue

        # Create a perturbed version of the text by masking the current word
        perturbed_text = " ".join(words[:i] + ["[MASK]"] + words[i+1:])

        # Classify the sentiment of the perturbed text
        perturbed_scores = classifier(perturbed_text)[0]

        # Calculate the difference in sentiment scores
        score_diff = {sentiment['label']: abs(original_score['score'] - sentiment['score'])
                      for original_score, sentiment in zip(original_scores, perturbed_scores)}

        # Store the score difference
        importance_scores[word] = score_diff

    return importance_scores

# Create All Sentiment Word Importance Visualization

def all_sent_importance_viz(sentiment_colors, lyrics, importance_scores):
    # Calculate opacity levels
    min_opacity = int(255 * 0.10)
    max_opacity = 255
    mid_opacity = (max_opacity + min_opacity) // 2

    # Generate HTML content for lyrics
    html_content = "<div style='font-size:16px;'>"

    # Variables to hold max and min scores for each sentiment
    max_scores = {sentiment: float('-inf') for sentiment in sentiment_colors}
    min_scores = {sentiment: float('inf') for sentiment in sentiment_colors}

    for word in lyrics.split():
        # Remove punctuation for lookup
        clean_word = word.strip('()[]:,.?!\"')
        # Get the sentiment scores for the clean word
        scores = importance_scores.get(clean_word, {})

        if not scores:
            continue

        for sentiment, score in scores.items():
            # Update max and min scores for the sentiment
            max_scores[sentiment] = max(max_scores[sentiment], score)
            min_scores[sentiment] = min(min_scores[sentiment], score)

        # Find the sentiment with the highest contribution for the word
        highest_sentiment = max(scores, key=scores.get)
        # Get the sentiment score for the chosen sentiment
        highest_sentiment_score = scores[highest_sentiment]

        # Normalize the sentiment score to [0, 1]
        max_score = max_scores[highest_sentiment]
        min_score = min_scores[highest_sentiment]
        normalized_score = (highest_sentiment_score - min_score) / (max_score - min_score) if max_score != min_score else 0.5

        # Calculate opacity proportionately between min and max
        opacity = int(min_opacity + (normalized_score * (max_opacity - min_opacity)))

        # Ensure opacity is within bounds
        opacity = max(min(opacity, 255), 0)

        # Combine color and opacity
        highlight_color_with_opacity = f"{sentiment_colors[highest_sentiment]}{opacity:02X}"
        background_style = f"background-color:{highlight_color_with_opacity};"
        # Append the word with the corresponding highlighting
        html_content += f"<span style='{background_style}'>{word} </span>"

    html_content += "</div>"

    # Generate HTML content for legend
    legend_html_content = "<div style='font-size:14px; margin-top:10px;'>"
    # Sort sentiments by max score in descending order
    sorted_sentiments = sorted(max_scores.keys(), key=lambda x: max_scores[x], reverse=True)
    for index, sentiment in enumerate(sorted_sentiments, start=1):
        max_score = max_scores[sentiment]
        min_score = min_scores[sentiment]
        # Set sentiment name in bold with a number before it
        legend_html_content += f"<b>{index}. {sentiment.capitalize()}</b><br>"
        # Highlight the maximum value
        max_highlight_color_with_opacity = f"{sentiment_colors[sentiment]}{max_opacity:02X}"
        max_background_style = f"color:black; background-color:{max_highlight_color_with_opacity}; opacity: {max_opacity:.2f}"
        legend_html_content += f"<span style='{max_background_style}'>Maximum value: {max_score:.6f}</span><br>"
        # Highlight the minimum value
        min_highlight_color_with_opacity = f"{sentiment_colors[sentiment]}{min_opacity:02X}"
        min_background_style = f"color:black; background-color:{min_highlight_color_with_opacity}; opacity: {min_opacity:.2f}"
        legend_html_content += f"<span style='{min_background_style}'>Minimum value: {min_score:.6f}</span><br><br>"

    legend_html_content += "</div>"

    return html_content, legend_html_content

# Create Sentiment-Specific Word Importance Visualization

def sent_specific_importance_viz(lyrics, chosen_sentiment, importance_scores, sentiment_colors):
    # Extract scores for the chosen sentiment for each word
    chosen_sentiment_scores = {word: float(scores[chosen_sentiment]) for word, scores in importance_scores.items()}

    # Get maximum, minimum, and middle values
    max_score = max(chosen_sentiment_scores.values())
    min_score = min(chosen_sentiment_scores.values())
    mid_score = (max_score + min_score) / 2

    # Normalize the scores to [0, 1] for color intensity
    normalized_scores = {word: (score - min_score) / (max_score - min_score) if max_score != min_score else mid_score for word, score in chosen_sentiment_scores.items()}

    # Calculate opacity levels for legend
    min_opacity = int(255 * 0.10)
    max_opacity = 255
    mid_opacity = (max_opacity + min_opacity) // 2

    # Generate HTML content for lyrics
    html_content = "<div style='font-size:16px;'>"
    for word in lyrics.split():
        # Remove punctuation for lookup
        clean_word = word.strip(',.?!\"')
        normalized_score = normalized_scores.get(clean_word, 0)
        # Get the color for the sentiment
        highlight_color = sentiment_colors[chosen_sentiment]
        # Calculate opacity proportionately between min and max
        opacity = int( min_opacity + (normalized_score * (max_opacity - min_opacity)))
        # Ensure opacity is within bounds
        opacity = max(min(opacity, 255), 0)
        # Combine color and opacity
        highlight_color_with_opacity = f"{highlight_color}{opacity:02X}"  # Convert opacity to hexadecimal
        background_style = f"background-color:{highlight_color_with_opacity};"
        # Append the word with the corresponding highlighting
        html_content += f"<span style='{background_style}'>{word} </span>"
    html_content += "</div>"

    # Generate HTML content for legend
    legend_html_content = f"<div style='font-size:14px; margin-top:10px;'>"
    legend_html_content += "<span style='margin-right: 10px;'>Minimum value</span>"
    legend_html_content += f"<span style='background-color:{sentiment_colors[chosen_sentiment]}{min_opacity:02X};'>{min_score:.6f}</span>"
    legend_html_content += "<span style='margin-left: 10px; margin-right: 10px;'>Middle value</span>"
    legend_html_content += f"<span style='background-color:{sentiment_colors[chosen_sentiment]}{mid_opacity:02X};'>{mid_score:.6f}</span>"
    legend_html_content += "<span style='margin-left: 10px; margin-right: 10px;'>Maximum value</span>"
    legend_html_content += f"<span style='background-color:{sentiment_colors[chosen_sentiment]}{max_opacity:02X};'>{max_score:.6f}</span>"
    legend_html_content += "</div>"

    return html_content, legend_html_content
    # Display the colored text and legend
    #display(HTML(html_content))
    #display(HTML(legend_html_content))

'''
# Obtain & Display Word-Specific Importance Values for the Selected Sentiment

def inspect_word_sentiment(word, sentiment_colors):
    
    # Extract scores for the chosen sentiment for each word
    chosen_sentiment_scores = {word: float(scores[chosen_sentiment]) for word, scores in importance_scores.items()}

    # Get maximum, minimum, and middle values
    max_score = max(chosen_sentiment_scores.values())
    min_score = min(chosen_sentiment_scores.values())
    mid_score = (max_score + min_score) / 2

    # Normalize the scores to [0, 1] for color intensity
    normalized_scores = {word: (score - min_score) / (max_score - min_score) if max_score != min_score else mid_score for word, score in chosen_sentiment_scores.items()}

    # Extract Word Values
    score = chosen_sentiment_scores.get(word, 0)
    normalized_score = normalized_scores.get(word, 0)
    opacity = int( min_opacity + (normalized_score * (max_opacity - min_opacity)))
    highlight_color = sentiment_colors[chosen_sentiment]
    highlight_color_with_opacity = f"{highlight_color}{opacity:02X}"

    # Generate HTML content to display the word with highlighting color and opacity
    html_content = f"<div style='font-size:16px;'>"
    background_style = f"background-color:{highlight_color_with_opacity};"
    html_content += f"<span style='{background_style}'>{word}</span>"
    html_content += "</div>"

    # Display the HTML content
    display(HTML(html_content))
    print(f"Sentiment Score: {score}")
    print(f"Normalized Score: {normalized_score}")
    print(f"opacity: {opacity}")
    print(f"Highlighting Color: {highlight_color_with_opacity}")
'''

# Display Sentiment-Specific Word Importance Histogram

def set_specific_importance_histogram(importance_scores, chosen_sentiment, sentiment_colors):
    # Get the sentiment scores for all words
    sentiment_values = [scores[chosen_sentiment] for scores in importance_scores.values()]

    # Create histogram trace with 20 bins
    fig = go.Figure(data=[go.Histogram(x=sentiment_values, nbinsx=20, marker_color=sentiment_colors[chosen_sentiment])])

    # Update layout
    fig.update_layout(
        title=f'Distribution of Sentiment Values for {chosen_sentiment.capitalize()} Sentiment',
        xaxis_title='Sentiment Value',
        yaxis_title='Frequency',
        bargap=0.05,  # Gap between bars
        xaxis=dict(
            tickfont=dict(size=10),  # Font size of tick labels
            tickangle=45,  # Rotate tick labels by 45 degrees
            tickformat=".6f"  # Format tick labels to display six decimal places
        )
    )
    return fig
    # Show plot
    #fig.show()

'''
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

def chunk_text(text, max_chunk_size, overlap_size, tokenizer):
    tokens = tokenizer.tokenize(text)
    max_chunk_size -= 2  # Adjust for special tokens
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = start_idx + max_chunk_size
        if end_idx > len(tokens): end_idx = len(tokens)
        chunk = [tokenizer.cls_token] + tokens[start_idx:end_idx] + [tokenizer.sep_token]
        chunk_str = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_str)
        start_idx = end_idx - overlap_size
    return chunks

def average_sentiment(text, tokenizer, classifier, max_chunk_size=512, overlap_size=50):
    chunks = chunk_text(text, max_chunk_size, overlap_size, tokenizer)
    total_scores = [0] * 7  # Assuming 7 sentiment categories
    num_chunks = len(chunks)
    for chunk in chunks:
        try:
            scores = classifier(chunk)
            for i, score in enumerate(scores[0]):
                total_scores[i] += score['score']
        except Exception as e:
            print(f"An error occurred: {e}")
    average_scores = [score / num_chunks for score in total_scores]
    return average_scores

def extract_keywords(lyrics_series):
    text = ' '.join(lyrics_series.dropna())
    text = re.sub(r'[^\w\s]', '', text)
    words = text.lower().split()
    word_counts = Counter(words)
    return word_counts.most_common(10)

def map_token_scores_to_words(tokens, token_scores, original_words):
    word_scores = {}
    for i, word in enumerate(original_words):
        word_scores[word] = token_scores[i]
    return word_scores

def sent_specific_importance_viz(original_words, word_scores, sentiment_colors, importance_scores, selected_sentiment):
    highlighted_lyrics = "<div style='white-space: pre-wrap;'>"
    for word in original_words:
        score = word_scores.get(word, 0.5)
        sentiment = selected_sentiment.lower()
        color = sentiment_colors.get(sentiment, '#FFFFFF')
        opacity = (score - 0.5) * 2
        background_color = f"rgba({color[1:]},{opacity:.2f})"
        highlighted_lyrics += f"<span style='background-color:{background_color};'>{word} </span>"
    highlighted_lyrics = re.sub(r'\s+', ' ', highlighted_lyrics)
    highlighted_lyrics += "</div>"
    return highlighted_lyrics

def normalize_scores(scores):
    max_score = max(scores.values()) if scores else 0
    min_score = min(scores.values()) if scores else 0
    return {word: (score - min_score) / (max_score - min_score) if max_score > min_score else 0.5 for word, score in scores.items()}

def create_and_display_pie_chart(sentiments, sentiment_values, colors):
    fig = go.Figure(data=[go.Pie(labels=sentiments, values=sentiment_values, marker=dict(colors=list(colors)))])
    fig.update_traces(textinfo='percent+label')
    return fig

def process_text_input(multiline_text):
    lines = multiline_text.split('\n')
    single_line_text = ' '.join(lines)
    return single_line_text

def word_importance_scores(text, classifier, tokenizer):
    text = text.replace('#', '')
    text = re.sub(r'\s+', ' ', text)
    tokens = tokenizer.tokenize(text)
    importance_scores = {}
    for i in range(0, len(tokens), tokenizer.model_max_length):
        chunk_tokens = tokens[i:i + tokenizer.model_max_length]
        original_scores = classifier(chunk_tokens, return_all_scores=True)[0]
        for j, token in enumerate(chunk_tokens):
            if token not in tokenizer.special_tokens_map.values():
                perturbed_chunk = chunk_tokens[:j] + [tokenizer.mask_token] + chunk_tokens[j + 1:]
                perturbed_text = tokenizer.convert_tokens_to_string(perturbed_chunk)
                perturbed_scores = classifier(perturbed_text, return_all_scores=True)[0]
                score_diff = {score['label']: abs(original_score['score'] - score['score']) for original_score, score in zip(original_scores, perturbed_scores)}
                importance_scores[token] = score_diff
    return importance_scores

'''



