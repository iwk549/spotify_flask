from flask import Flask, request, jsonify
from flask_cors import CORS
import playlist_functions as play
import json
import os

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True


@app.route('/api/v1/healthz', methods={'GET'})
def health_check():
    return "<p>Healthy</p>"


@app.route('/api/v1/spotify/artist', methods=["GET"])
def api_spotify_artist():
    e = play.set_spotipy_environment_variables(request.args)
    if e:
        return e, 400
    if 'id' in request.args:
        id = request.args['id']
    else:
        return 'Error: No id field provided. Please specify an artist id.', 400
    artist = play.get_artist_name(id)
    if 'Error' in artist:
        return artist, 400
    else:
        return json.dumps(artist)


@app.route('/api/v1/spotify/playlist', methods=["GET"])
def api_spotify_playlist():
    e = play.set_spotipy_environment_variables(request.args)
    if e:
        return e, 400
    if 'id' in request.args:
        id = request.args['id']
    else:
        return 'Error: No id field provided. Please specify an artist id.', 400
    playlist = play.get_playlist_name(id)
    if 'Error' in playlist:
        return playlist, 400
    else:
        return json.dumps(playlist)


@app.route('/api/v1/spotify/audio_features/artists/top_tracks', methods=['GET'])
def api_spotify_audio_features_artists_top_tracks():
    e = play.set_spotipy_environment_variables(request.args)
    if e:
        return e, 400
    if 'id' in request.args:
        id = request.args['id']
    else:
        return 'Error: No id field provided. Please specify an artist id.', 400
    audio_features = play.get_artist_top_tracks_audio_features(id)
    if audio_features is not None:
        return audio_features.to_json(orient='records')
    else:
        return 'Error: Invalid track id provided.', 400


@app.route('/api/v1/spotify/audio_features/artist', methods=['GET'])
def api_spotify_audio_features_artist():
    e = play.set_spotipy_environment_variables(request.args)
    if e:
        return e, 400
    if 'id' in request.args:
        id = request.args['id']
    else:
        return 'Error: No id field provided. Please specify an artist id.', 400
    audio_features = play.get_all_tracks_for_artist_audio_features(id)
    if audio_features is not None:
        return audio_features.to_json(orient='records')
    else:
        return 'Error: Invalid artist id provided.', 400


@app.route('/api/v1/spotify/audio_features/playlists', methods=['GET'])
def api_spotify_audio_features_artists():
    e = play.set_spotipy_environment_variables(request.args)
    if e:
        return e, 400
    if 'ids' in request.args:
        ids = request.args['ids']
    else:
        return 'Error: No id field provided. Please specify a track id.', 400
    audio_features = play.get_playlist_tracks_audio_features(ids.split(','))
    if audio_features is not None:
        return audio_features.to_json(orient='records')
    else:
        return 'Error: Invalid playlist id provided.', 400


@app.route('/api/v1/spotify/audio_features/playlist_recommendations', methods=['GET'])
def api_spotify_playlist_recomendations():
    e = play.set_spotipy_environment_variables(request.args)
    if e:
        return e, 400
    if 'artistid' not in request.args or 'playlistids' not in request.args:
        return 'Error: Incorrect parameters supplied.', 400

    artist_id = request.args['artistid']
    playlist_ids = request.args['playlistids'].split(',')
    probability = False if request.args['probability'] == 'false' else True
    all_songs = False if request.args['all_songs'] == 'false' else True
    
    playlist_tracks = play.get_playlist_tracks_audio_features(playlist_ids)
    if all_songs:
        artist_tracks = play.get_all_tracks_for_artist_audio_features(artist_id)
    else:
        artist_tracks = play.get_artist_top_tracks_audio_features(artist_id)

    if 'Error' in playlist_tracks:
        return playlist_tracks, 400
    if 'Error' in artist_tracks:
        return artist_tracks, 400

    prediction_columns = [
        'danceability',
        'energy',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence'
    ]
    result_column = ['playlist_name']

    if len(playlist_ids) == 1:
        predictions = play.find_best_fitting_song(playlist_tracks, artist_tracks, prediction_columns)
        return {'predictions': predictions.to_json(orient='records')}

    kernel, gamma, c, svc_score = \
        play.find_best_estimators(
            playlist_tracks, prediction_columns, result_column, estimator='svc')
    k, knn_score = \
        play.find_best_estimators(
            playlist_tracks, prediction_columns, result_column, estimator='knn')

    estimator = {
        'model': 'svc',
        'kernel': kernel,
        'C': c,
        'gamma': gamma
    }
    if knn_score > svc_score and not probability:
        estimator = {
            'model': 'knn',
            'k': k
        }

    if probability:
        predictions, xx, yy, Z = \
            play.playlist_probabilities(playlist_tracks, artist_tracks, estimator,
                                        prediction_columns, result_column)
    else:
        predictions = \
            play.find_best_fitting_playlist(playlist_tracks, artist_tracks, estimator,
                                            prediction_columns, result_column, race=False)
        xx = ''
        yy = ''
        Z = ''

    if xx != '' and yy != '' and Z != '':
        y = []
        yy = yy.tolist()
        for each in yy:
            y.append(each[0])
        xx = xx.tolist()
        x = xx[0]
        x = json.dumps(x)
        y = json.dumps(y)
        Z = json.dumps(Z.tolist())
    else:
        x = 'null'
        y = 'null'
        Z = 'null'

    return {
        'predictions': predictions.to_json(orient='records'),
        'xx': x,
        'yy': y,
        'Z': Z
    }


if __name__ == '__main__':
    app.run()
