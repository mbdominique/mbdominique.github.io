import os
import json

from kaggle.api.kaggle_api_extended import KaggleApi

from dagster import asset, Field, repository, make_values_resource, Permissive, In, Out, AssetKey

import pandas as pd

import numpy as np

@asset(config_schema={
    'dataset_name': Field(str, description="Name of the Kaggle dataset to download. Format should be 'username/dataset'."),
    'download_path': Field(str, default_value='./data', description="Path to save the downloaded dataset.")
})
def pull_kaggle_dataset(context) -> pd.DataFrame:
    dataset_name = context.op_config['dataset_name']
    download_path = context.op_config['download_path']

    os.makedirs(download_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset_name, path=download_path, unzip=True)

    csv_file = next((f for f in os.listdir(download_path) if f.endswith('.csv')), None)
    if not csv_file:
        raise Exception(f"No CSV file found in the downloaded dataset {dataset_name}.")

    csv_file_path = os.path.join(download_path, csv_file)
    df = pd.read_csv(csv_file_path)

    context.log.info(f"Dataset {dataset_name} loaded into DataFrame from {csv_file_path}")
    init_df = df
    return init_df

@asset(ins={"init_df": In(asset_key=AssetKey("pull_kaggle_dataset"))})
def clean_data(init_df: pd.DataFrame) -> pd.DataFrame:
    init_df.columns = init_df.columns.str.lower().str.replace(' ', '_')
    
    for col in init_df.select_dtypes(include=['float', 'int']).columns:
        init_df[col].fillna(init_df[col].median(), inplace=True)
    
    for col in init_df.select_dtypes(include=['object', 'category']).columns:
        init_df[col].fillna(init_df[col].mode()[0], inplace=True)
    
    if 'date' in init_df.columns:
        init_df['date'] = pd.to_datetime(init_df['date'])
    
    clean_df = init_df

    return clean_df

@asset(ins={"clean_df": In(asset_key="clean_data")})
def feature_engineering(clean_df: pd.DataFrame) -> pd.DataFrame:
    if 'artist(s)_name' in clean_df.columns:
        artists_split = clean_df['artist(s)_name'].str.split(', ', expand=True)
        for i in range(artists_split.shape[1]):
            clean_df[f'Artist_{i+1}'] = artists_split[i]
        clean_df.drop(columns=['artist(s)_name'], inplace=True)

    if 'streams' in clean_df.columns:
        clean_df['streams'] = pd.to_numeric(clean_df['streams'], errors='coerce')
        clean_df.dropna(subset=['streams'], inplace=True)
        clean_df['streams'] = clean_df['streams'].astype(int)

    date_columns = ['released_year', 'released_month', 'released_day']
    if all(col in clean_df.columns for col in date_columns):
        clean_df['release_date'] = pd.to_datetime(clean_df[date_columns].astype(str).agg('-'.join, axis=1), errors='coerce')
        clean_df.drop(columns=date_columns, inplace=True)

    feat_df = clean_df
    
    return feat_df

@asset(
    config_schema={
        'song_name': Field(str, is_required=False, description="The name of the song to base the playlist on."),
        'genre': Field(str, is_required=False, description="Filter for the genre of songs in the playlist."),
        'artist': Field(str, is_required=False, description="Filter for the artist of songs in the playlist."),
        'year_range': Field(Permissive({'start_year': int, 'end_year': int}), is_required=False, description="Filter for the release year range of songs in the playlist."),
        'num_songs': Field(int, default_value=10, description="Number of songs to include in the playlist.")
    },
    ins={"feat_df": In(asset_key="feature_engineering")}
)
def playlist_creator(context, feat_df: pd.DataFrame) -> pd.DataFrame:
    song_name = context.op_config.get('song_name')
    genre = context.op_config.get('genre')
    artist = context.op_config.get('artist')
    year_range = context.op_config.get('year_range')
    num_songs = context.op_config.get('num_songs')

    playlist = feat_df

    if song_name:
        base_song = feat_df[feat_df['track_name'].str.lower() == song_name.lower()]
        if not base_song.empty:
            base_song = base_song.iloc[0]
            genre = genre or base_song['genre']
            artist = artist or base_song['artist']
            if not year_range:
                year = base_song.get('year')
                year_range = {'start_year': year, 'end_year': year}

    if genre:
        playlist = playlist[playlist['genre'].str.lower() == genre.lower()]
    if artist:
        playlist = playlist[playlist['artist'].str.lower() == artist.lower()]

    if year_range:
        playlist = playlist[(playlist['year'] >= year_range['start_year']) & (playlist['year'] <= year_range['end_year'])]

    playlist = playlist.sort_values(by='year', ascending=False).head(num_songs)

    return playlist

@asset(ins={"feat_df": In(asset_key="feature_engineering")})
def cross_analysis(context, feat_df: pd.DataFrame) -> dict:
    song = context.op_config['song']
    if 'track_name' not in feat_df.columns:
        raise ValueError("DataFrame does not contain 'track_name' column.")

    song_data = feat_df[feat_df['track_name'].str.lower() == song.lower()]

    streams_data = {}

    if song_data.empty:
        context.log.warning(f"Song '{song}' not found in the dataset.")
        return streams_data 

    for col in feat_df.columns:
        if 'streams' in col.lower():  
            platform_name = col.replace('_', ' ').title()  
            streams_count = song_data[col].values[0] if not song_data[col].isna().all() else None
            streams_data[platform_name] = streams_count

    return streams_data


# LOAD

@asset(ins={"feat_df": In(asset_key="feature_engineering")})
def load_data_to_csv(context, feat_df: pd.DataFrame) -> str:
    csv_file_path = 'data/transformed_kaggle_dataset.csv'
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    feat_df.to_csv(csv_file_path, index=False)
    context.log.info(f"Data successfully saved to {csv_file_path}")
    return csv_file_path


@asset(
    config_schema={
        'playlist_csv_file_path': Field(str, description="File path for saving the playlist DataFrame as CSV."),
        'streams_data_json_file_path': Field(str, description="File path for saving the streams data dictionary as JSON."),
        'playlist_json_file_path': Field(str, description="File path for saving the playlist data dictionary as JSON.")
    }
)
def save_data(context, playlist: pd.DataFrame, streams_data: dict):
    playlist_csv_file_path = context.op_config['playlist_csv_file_path']
    streams_data_json_file_path = context.op_config['streams_data_json_file_path']
    playlist_json_file_path = context.op_config['playlist_json_file_path']

    playlist.to_csv(playlist_csv_file_path, index=False)
    context.log.info(f"Playlist DataFrame saved to {playlist_csv_file_path}")

    with open(streams_data_json_file_path, 'w') as f:
        json.dump(streams_data, f, indent=4)
    context.log.info(f"Streams data saved to {streams_data_json_file_path}")

    with open(playlist_json_file_path, 'w') as f:
        json.dump(playlist.to_dict(orient='records'), f, indent=4)
    context.log.info(f"Playlist data saved to {playlist_json_file_path}")


my_project_defs = Definitions(
    assets=[
        pull_kaggle_dataset,
        clean_data,
        feature_engineering,
        playlist_creator,
        cross_analysis,
        load_data_to_csv,
        save_data,
    ]
)

@repository
def my_repository():
    """
    A Dagster repository that gathers all the assets for the data pipeline.
    """
    return [my_project_defs]