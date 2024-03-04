import streamlit as st
from utils import df

def show():
    st.header("Playlist Finder")
    st.write('Filters')
    col1, col2 = st.columns(2)

    with col1:
        music_type = st.selectbox("By Music Type:", ['All'] + sorted(df['track_genre'].unique().tolist()))

    with col2:
        keyword = st.text_input('Keyword (in playlist name)')

    if st.button('Get Results'):
        if music_type != 'All':
            # Corrected to avoid regex interpretation issues
            filtered_df = df[df['track_genre'].str.contains(music_type, case=False, na=False, regex=False)]
        else:
            filtered_df = df.copy()

        if keyword:
            # Corrected to avoid regex interpretation issues
            filtered_df = filtered_df[filtered_df['playlist.name'].str.contains(keyword, case=False, na=False, regex=False)]

        if not filtered_df.empty:
            grouped_df = filtered_df.groupby('playlist.name').agg({
                'track_genre': lambda x: ', '.join(x.unique()),
                'artist_name': lambda x: ', '.join(sorted(x.unique())),
                'track.name': lambda x: ', '.join(sorted(x.unique()))
            }).reset_index()

            display_df = grouped_df.rename(columns={
                'playlist.name': 'Playlist Name',
                'track_genre': 'Genre',
                'artist_name': 'Artists Name',
                'track.name': 'Songs'
            })

            st.write("Filtered Playlists:")
            st.dataframe(display_df[['Playlist Name', 'Genre', 'Artists Name']])

            selected_playlists = st.multiselect('Select Playlists to View Songs', options=display_df['Playlist Name'].tolist())

            if selected_playlists:
                st.write("Selected Playlists Details")
                for playlist in selected_playlists:
                    playlist_details = display_df[display_df['Playlist Name'] == playlist]
                    st.write(f"**{playlist}** - Songs:")
                    songs = playlist_details['Songs'].values[0].split(', ')
                    st.write(', '.join(songs))
        else:
            st.write('No playlists match your criteria.')

        st.write('---')

