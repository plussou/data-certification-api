
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint

@app.get("/predict")
def predict(acousticness, danceability, duration_ms, energy, explicit, id, \
            instrumentalness, key, liveness, loudness, mode, name, \
            release_date, speechiness, tempo, valence, artist):


    X = pd.DataFrame( columns = ['acousticness', 'danceability', 'duration_ms',  \
                                 'energy', 'explicit', 'id','instrumentalness', \
                                  'key', 'liveness', 'loudness', 'mode','name', \
                                  'release_date', 'speechiness', 'tempo', \
                                  'valence', 'artist', 'year'\
                                  ])

    X.loc[0,'acousticness'] = float(acousticness)
    X.loc[0,'danceability'] = float(danceability)
    X.loc[0,'duration_ms'] = float(duration_ms)
    X.loc[0,'energy'] = float(energy)
    X.loc[0,'explicit'] = float(explicit)
    X.loc[0,'id'] = id
    X.loc[0,'instrumentalness'] = float(instrumentalness)
    X.loc[0,'key'] = float(key)
    X.loc[0,'liveness'] = float(liveness)
    X.loc[0,'loudness'] = float(loudness)
    X.loc[0,'mode'] = float(mode)
    X.loc[0,'name'] = name
    X.loc[0,'release_date'] = release_date
    X.loc[0,'speechiness'] = float(speechiness)
    X.loc[0,'tempo'] = float(tempo)
    X.loc[0,'valence'] = float(valence)
    X.loc[0,'artist'] = artist

#    release_year = pd.to_datetime(X['release_date']).dt.year
#    X.loc[0,'year'] = release_year.values[0]
#    X.drop(columns=['release_date'],inplace=True)

    pipeline = joblib.load('model.joblib')
    y=pipeline.predict(X)

    return {'predict':y[0]}
