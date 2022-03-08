import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOauthError
import requests.exceptions as re
import spotipy.exceptions as se
import pprint


def set_spotipy_environment_variables(args):
    # Client IDs set as env variables
    pass


def login():
    token = spotipy.oauth2.SpotifyClientCredentials().get_access_token()
    print(token)
    return spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())


def get_artist_name(artist_id):
    spotify = login()
    try:
        artist = spotify.artist(artist_id)
    except (re.HTTPError, se.SpotifyException) as e:
        return "Error: An invalid artist id was provided."
    except SpotifyOauthError:
        return 'Error: Invalid Login Credentials.'
    return {
        'name': artist['name'],
        'followers': artist['followers']['total'],
        'image': artist['images'][0]['url'],
        'genres': artist['genres'],
    }


def get_playlist_name(playlist_id):
    spotify = login()
    try:
        # playlist = spotify.playlist(playlist_id, fields='followers,name,owner,tracks')
        playlist = spotify.playlist(playlist_id)
    except (re.HTTPError, se.SpotifyException) as e:
        return "Error: An invalid playlist id was provided."
    except SpotifyOauthError:
        return 'Error: Invalid Login Credentials.'
    return {
        'name': playlist['name'],
        'followers': playlist['followers']['total'],
        'owner': playlist['owner']['display_name'],
        'owner_id': playlist['owner']['id']
    }