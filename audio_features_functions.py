import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import svm, preprocessing
from sklearn.neighbors import KNeighborsClassifier
import requests.exceptions as re
import spotipy.exceptions as se
from spotipy.oauth2 import SpotifyOauthError
from statistics import mean
from scipy.spatial import distance

import spotify_functions


def destructure_track_info(track):
    track_info = {}
    artists = ''
    for artist in track['artists']:
        artists += artist['name'] + ', '
    artists = artists[:-2]
    track_info['artists'] = artists
    track_info['name'] = track['name']
    track_info['id'] = track['id']
    track_info['album_name'] = track['album']['name']
    track_info['album_id'] = track['album']['id'],
    track_info['spotify_link'] = track['external_urls']['spotify']
    return track_info


def destructure_audio_features(audio_features, all_tracks, extra_param='playlist'):
    complete_track_list = []
    for track in audio_features:
        for each_track in all_tracks:
            try:
                if each_track['id'] == track['id']:
                    track['name'] = each_track['name']
                    track['artists'] = each_track['artists']
                    track[extra_param + '_name'] = each_track[extra_param + '_name']
                    track[extra_param + '_id'] = each_track[extra_param + '_id']
                    try:
                        track['spotify_link'] = each_track['spotify_link']
                    except KeyError:
                        pass
                    complete_track_list.append(track)
            except TypeError:
                pass
    return pd.DataFrame(complete_track_list)


def split_list(original_list, length=100):
    x = 0
    new_list = []
    while x < len(original_list):
        new_list.append(original_list[x:x+length])
        x += length
    return new_list


def get_playlist_tracks_audio_features(playlist_array):
    """
    Get all the tracks from the provided playlists with audio features
    :param playlist_array: an array of playlist ids
    :return: pandas DataFrame of all tracks and audio features
    """
    spotify = spotify_functions.login()
    all_tracks = []
    audio_features = []
    for playlist_id in playlist_array:
        try:
            data = spotify.playlist(playlist_id)
        except (re.HTTPError, se.SpotifyException) as e:
            return "Error: An invalid playlist id was provided."
        except SpotifyOauthError:
            return 'Error: Invalid Login Credentials.'

        track_ids = []
        for track in data['tracks']['items']:
            track_info = {}
            artists = ''
            try:
                for artist in track['track']['artists']:
                    artists += artist['name'] + ', '
                artists = artists[:-2]
                track_info['artists'] = artists
                track_info['name'] = track['track']['name']
                track_info['id'] = track['track']['id']
                track_info['playlist_name'] = data['name']
                track_info['playlist_id'] = playlist_id
                all_tracks.append(track_info)
                track_ids.append(track_info['id'])
            except TypeError:
                pass
        split_ids = split_list(track_ids)
        for each in split_ids:
            audio_features.extend(spotify.audio_features(each))

    return destructure_audio_features(audio_features, all_tracks, 'playlist')


def get_all_tracks_for_artist_audio_features(artist_id):
    spotify = spotify_functions.login()
    try:
        albums = spotify.artist_albums(artist_id)
    except (re.HTTPError, se.SpotifyException) as e:
        return 'Error: Invalid artist id.'
    except SpotifyOauthError:
        return 'Error: Invalid Login Credentials.'

    track_ids = []
    all_tracks = []
    album_ids = []
    for album in albums['items']:
        if album['album_group'] != 'appears_on':
            album_ids.append({'album_id': album['id'], 'album_name': album['name']})
    try:
        for album in album_ids:
            album_tracks = spotify.album_tracks(album['album_id'])
            for track in album_tracks['items']:
                track['album'] = {'name': album['album_name']}
                track['album']['id'] = album['album_id']
                try:
                    track_info = destructure_track_info(track)
                    all_tracks.append(track_info)
                    track_ids.append(track_info['id'])
                except TypeError:
                    pass
    except (re.HTTPError, se.SpotifyException) as e:
        return 'Error: Invalid album id.'
    except SpotifyOauthError:
        return 'Error: Invalid Login Credentials.'

    audio_features = pd.DataFrame()
    split_ids = split_list(track_ids)
    for each in split_ids:
        audio_features = \
            audio_features.append(
                destructure_audio_features(
                    spotify.audio_features(each), all_tracks, 'album'), ignore_index=True)
    return audio_features


def get_artist_top_tracks_audio_features(artist_id):
    """
    Get all the tracks from the provided artist with audio features
    :param artist_id: a single artist id
    :return: pandas DataFrame of top tracks and audio features
    """
    spotify = spotify_functions.login()
    try:
        top_tracks = spotify.artist_top_tracks(artist_id)
    except (re.HTTPError, se.SpotifyException) as e:
        return 'Error: Invalid artist id.'
    except SpotifyOauthError:
        return 'Error: Invalid Login Credentials.'

    track_ids = []
    all_tracks = []
    for track in top_tracks['tracks']:
        try:
            track_info = destructure_track_info(track)
            all_tracks.append(track_info)
            track_ids.append((track_info['id']))
        except TypeError:
            pass

    audio_features = spotify.audio_features(track_ids)
    return destructure_audio_features(audio_features, all_tracks, 'album')


def get_single_track_audio_features(track_id):
    spotify = spotify_functions.login()
    try:
        track = spotify.track(track_id)
    except (re.HTTPError, se.SpotifyException) as e:
        return 'Error: Invalid track id'
    except SpotifyOauthError:
        return 'Error: Invalid Login Credentials.'

    try:
        all_tracks = [destructure_track_info(track)]
    except TypeError:
        raise Exception('Not enough info available for this track')

    audio_features = spotify.audio_features(track_id)
    return destructure_audio_features(audio_features, all_tracks, 'album')


def find_best_estimators(training_data, x_columns, y_column, estimator='svc',
                         cross_validation_count=5):
    """
    Use cross validation to get the best estimators for both SVC and KNN functions
    :param training_data: pandas DataFrame of complete data including y column
    :param x_columns: columns to use for prediction
    :param y_column: column to predict against (result)
    :param estimator: svc or knn, which model to use
    :param cross_validation_count: how many segments to split data into, default 5
    :return: best estimators and cross validation score.
            For SVC: kernel, C, gamma, cv score
            For KNN: k, cv score
    """
    # Initialize parameter grids
    gscv_svc_param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    ]
    gscv_knn_param_grid = {'n_neighbors': np.arange(1, 26)}

    # shuffle training data for more accurate results
    training_data = training_data.sample(frac=1).reset_index(drop=True)

    # split data into X & y sets
    training_x = training_data[x_columns]
    training_y = training_data[y_column].values.ravel()

    # find optimal estimators using cross validation
    if estimator == 'svc':
        gscv = GridSearchCV(estimator=svm.SVC(), param_grid=gscv_svc_param_grid,
                            n_jobs=1, cv=cross_validation_count)
    elif estimator == 'knn':
        gscv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=gscv_knn_param_grid,
                            cv=cross_validation_count)
    else:
        raise Exception('Estimator must be either svc or knn.')

    # fit data and find estimators
    gscv.fit(training_x, training_y)
    try:
        kernel = gscv.best_estimator_.kernel
        gamma = gscv.best_estimator_.gamma
        c = gscv.best_estimator_.C
        cv_score = np.mean(cross_val_score(svm.SVC(kernel=kernel,
                                                   C=c,
                                                   gamma=gamma),
                                           training_x, training_y,
                                           cv=cross_validation_count))
        return kernel, gamma, c, cv_score
    except AttributeError:
        k = next(iter(gscv.best_params_.values()))
        cv_score = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k),
                                           training_x, training_y,
                                           cv=cross_validation_count))
        return k, cv_score


def get_contour_data(training_x, training_y, clf):
    X = training_x.to_numpy()
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    le = preprocessing.LabelEncoder()
    le.fit(training_y)
    clf.fit(training_x, le.transform(training_y))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    return (xx, yy, Z)


def find_best_fitting_playlist(training_data, testing_data, estimator,
                               x_columns, y_column, race=False):
    """
    find the single best fitting playlist
    :param training_data: a dataframe of audio features of tracks with playlists
    :param testing_data: a dataframe of audio features of track without playlist
    :param estimator: an object with model and estimator numbers to use
    :param x_columns: prediction columns
    :param y_column: result column
    :param race: can be used for a single track only, will pair up results and reace them off
                against one another, will usually return the same results
    :return: the testing_data input with prediction appended
    """
    training_x = training_data[x_columns]
    training_y = training_data[y_column].values.ravel()
    testing_x = testing_data[x_columns]
    xx, yy, Z = '', '', ''
    if estimator['model'] == 'svc':
        clf = svm.SVC(C=estimator['C'], kernel=estimator['kernel'],
                      gamma=estimator['gamma'])
        if not race:
            clf.fit(training_x, training_y)
            testing_data['playlist_recommendation'] = clf.predict(testing_x)

        else:
            if testing_data.shape[0] != 1:
                raise Exception('You can only use the race function with a single track.',
                                'For multiple tracks use the basic prediction')
            race_frame = pd.DataFrame()
            for playlist, sub_group in training_data.groupby('playlist_name'):
                race_frame = race_frame.append(sub_group)
                if race_frame.groupby('playlist_name').nunique().shape[0] == 2:
                    clf.fit(race_frame[x_columns], race_frame[y_column].values.ravel())
                    prediction = clf.predict(testing_x)[0]
                    indexes_to_drop = \
                        race_frame[race_frame['playlist_name'] != prediction].index
                    race_frame.drop(indexes_to_drop, inplace=True)
            testing_data['playlist_recommendation'] = [prediction]

            xx, yy, Z = get_contour_data(training_x, training_y, clf)

            return [testing_data, xx, yy, Z]

    elif estimator['model'] == 'knn':
        knn = KNeighborsClassifier(n_neighbors=estimator['k'])
        if not race:
            knn.fit(training_x, training_y)
            testing_data['playlist_recommendation'] = knn.predict(testing_x)

        else:
            if training_data.shape[0] != 1:
                raise Exception('You can only use the race function with a single track.',
                                'For multiple tracks use the basic prediction')
            race_frame = pd.DataFrame()
            for playlist, sub_group in training_data.groupby('playlist_name'):
                race_frame = race_frame.append(sub_group)
                if race_frame.groupby('playlist_name').nunique().shape[0] == 2:
                    knn.fit(race_frame[x_columns], race_frame[y_column].values.ravel())
                    prediction = knn.predict(testing_x)[0]
                    indexes_to_drop = \
                        race_frame[race_frame['playlist_name'] != prediction].index
                    race_frame.drop(indexes_to_drop, inplace=True)
            testing_data['playlist_recommendation'] = [prediction]
            return testing_data

    else:
        raise Exception('Estimator model must be either svc or knn')

    return testing_data


def playlist_probabilities(training_data, testing_data, estimator, x_columns, y_column):
    training_x = training_data[x_columns]
    training_y = training_data[y_column].values.ravel()
    testing_x = testing_data[x_columns]
    if estimator['model'] == 'svc':
        clf = svm.SVC(C=estimator['C'], kernel=estimator['kernel'],
                      gamma=estimator['gamma'], probability=True)
        clf.fit(training_x, training_y)
        probabilities = clf.predict_proba(testing_x).transpose()
        x = 0
        for each in clf.classes_:
            testing_data[each] = probabilities[x]
            x += 1

    # xx, yy, Z = get_contour_data(training_x, training_y, clf)
    xx, yy, Z = "", "", ""
    return [testing_data, xx, yy, Z]


def find_best_fitting_song(playlist_audio_features, artist_song_audio_features, x_columns):

    features = list(zip(
        playlist_audio_features['danceability'].to_list(),
        playlist_audio_features['energy'].to_list(),
        playlist_audio_features['speechiness'].to_list(),
        playlist_audio_features['acousticness'].to_list(),
        playlist_audio_features['instrumentalness'].to_list(),
        playlist_audio_features['liveness'].to_list(),
        playlist_audio_features['valence'].to_list()
    ))

    euclidean_distances_column = []
    for i, row in artist_song_audio_features.iterrows():
        euclidean_distances = []
        for feature in features:
            euclidean_distance = \
                distance.euclidean(feature,
                                   [row['danceability'],
                                    row['energy'],
                                    row['speechiness'],
                                    row['acousticness'],
                                    row['instrumentalness'],
                                    row['liveness'],
                                    row['valence']])
            euclidean_distances.append(euclidean_distance)
        euclidean_distances_column.append(mean(euclidean_distances))
    artist_song_audio_features['euclidean_distance'] = euclidean_distances_column

    return artist_song_audio_features
