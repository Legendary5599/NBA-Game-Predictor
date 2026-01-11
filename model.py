
from flask import Flask, request, jsonify, render_template
from requests.exceptions import ReadTimeout
import joblib
import pandas as pd
import numpy as np
import time
from datetime import date
#Specific NBA Libraries
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamestimatedmetrics
from nba_api.stats.endpoints import TeamGameLogs
from nba_api.stats.endpoints import ScoreboardV2

#Help with API Timeouts
from nba_api.stats.endpoints import commonplayerinfo
headers = {
    #Put own headers
}

app = Flask(__name__)

# Load model and data at startup
model_xgb = joblib.load('models/xgb_model.pkl')
model_lgb = joblib.load('models/lgb_model.pkl')
scaler = joblib.load('models/scaler.pkl') 
feature_names = joblib.load('models/feature_names.pkl')
data = joblib.load('models/data.pkl')

#Get games happening that day
def games_today():
    today = date.today()
    game_date = today.strftime('%m/%d/%Y')
    print(f"Fetching games for: {game_date}")
    #call api
    board = ScoreboardV2(game_date=game_date)
    games_df = board.get_data_frames()[0]
    teams = games_df[['HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
    return teams

#translate id to abbr
def translate():
    abbr = []
    #Get all teams in NBA
    nba_teams = teams.get_teams()
    tm = pd.DataFrame(nba_teams)
    #translate id to abbr
    ids = games_today()
    for row in ids.itertuples():
        homeId = row.HOME_TEAM_ID
        awayId = row.VISITOR_TEAM_ID
        print(homeId, awayId)
        homeAbr = tm[tm['id'] == homeId]['abbreviation'].iloc[0]
        awayAbr = tm[tm['id'] == awayId]['abbreviation'].iloc[0]
        
        abbr.append((homeAbr, awayAbr))
        
    return abbr

#translate id to team name
def translateName():
    name = []
    #Get all teams in NBA
    nba_teams = teams.get_teams()
    tm = pd.DataFrame(nba_teams)
    #translate id to abbr
    ids = games_today()
    for row in ids.itertuples():
        homeId = row.HOME_TEAM_ID
        awayId = row.VISITOR_TEAM_ID
        print(homeId, awayId)
        homeAbr = tm[tm['id'] == homeId]['full_name'].iloc[0]
        awayAbr = tm[tm['id'] == awayId]['full_name'].iloc[0]
        
        name.append((homeAbr, awayAbr))
        
    return name

#number of games being played
def number():
    df = games_today()
    length = len(df)
    return length

#get rows that correlate to home team
def get_data():
    rows = []
    abbr = translate()
    for home, away in abbr:
        recent = data[data['HOME_ABBR'] == home].iloc[0]
        recent['AWAY_ABBR'] = away
        rows.append(recent)
    games = pd.DataFrame(rows)
    newGames = pd.concat([games, data], ignore_index=True)
    return newGames

#clean data for model
def clean():
    games = get_data()
    X = games.drop(columns = ['WL', 'WL_NUM'])
    col = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X[col] = scaler.fit_transform(X[col])
    X = pd.DataFrame(pd.get_dummies(X))
    a = number()
    X = X.head(a)
    return X

@app.route('/')
def home():
    """Main page - automatically makes prediction and displays it"""
    
    #Prediction logic
    games = []
    input_data = clean()
    name = translateName()
    ids = games_today()
    prediction = model_lgb.predict(input_data)
    
    for i in range(len(prediction)):
        homeName = name[i][0]
        awayName = name[i][1]
        
        homeAbbr = ids['HOME_TEAM_ID'].iloc[i]
        awayAbbr = ids['VISITOR_TEAM_ID'].iloc[i]
        if prediction[i] == 1:
            result = '<'
        else:
            result = '>'
            
        dict = {'home_team': homeName,
            'home_abbr': homeAbbr,
            'away_team': awayName, 
            'away_abbr': awayAbbr,
            'prediction': result }
        games.append(dict)
    
    return render_template('webpage.html', games=games)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON from the request body
        data = request.get_json()
        
        # Extract features - expecting a list or list of lists
        # Single prediction: {"features": [1.5, 2.3, 4.1, ...]}
        # Batch prediction: {"features": [[1.5, 2.3, ...], [2.1, 3.4, ...]]}
        features = data.get('features')
        
        if features is None:
            return jsonify({'error': 'Missing "features" in request'}), 400
        
        input_data = clean()
        # Make prediction
        predictions = model_lgb.predict(input_data)
        
        return jsonify({
            'predictions': predictions.tolist(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)