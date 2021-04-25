import pandas as pd
import numpy as np
import joblib
#import sklearn


#returns the avg economy rate, bowlers is a list or a np array
def get_avg_economy_rate(bowlers,bowlers_database_PP):
    
    if len(bowlers)==0:
        return 7.0
    
    economy_t = 0
    for bowler in bowlers:
        if bowler in bowlers_database_PP.Bowler.values:
            index_of_bowler = bowlers_database_PP[bowlers_database_PP.Bowler==bowler].index.values
            economy = float(bowlers_database_PP.Economy_Rate.iloc[index_of_bowler])
        else:
            economy = 7.0
           
        economy_t += economy
    return round(economy_t/len(bowlers),2)

#****************************************************
#function returns the average strike rate of the batsmen,batsmen is a list or ndarray of batsmen names

def get_avg_strike_rate(batsmen,batsmen_database_PP):
    
    if len(batsmen)==0:
        return 100
    strike_rate_t=0  #stores the total strike rate
    for batsman in batsmen:
        if batsman in batsmen_database_PP.Batsman.values:
            index_of_batsman = batsmen_database_PP[batsmen_database_PP.Batsman==batsman].index.values
            strike_rate = float(batsmen_database_PP.Strike_Rate.iloc[index_of_batsman])
            
        else:    #if the batsman is new, then assuming his strikerate is 100
            strike_rate = 100
        
        strike_rate_t += strike_rate
            
    return round(strike_rate_t/len(batsmen),2)


def predictRuns(testInputpath):
    prediction = 0
    venue_encoder = joblib.load('venue_encoder.joblib')
    team_encoder = joblib.load('team_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('linear_regression.joblib')
    batsmen_database_PP = pd.read_csv('batsmen_database_PP.csv')
    bowlers_database_PP = pd.read_csv('bowlers_database_PP.csv')
    testInput = pd.read_csv(testInputpath)
    
    
    venue = testInput.venue.values
    innings = testInput.innings.values[0]
    batting_team = testInput.batting_team
    bowling_team = testInput.bowling_team
    batsmen = testInput.batsmen[0].split(',')
    bowlers = testInput.bowlers[0].split(',')
    
    venue = venue_encoder.transform(venue)[0]
    batting_team = team_encoder.transform(batting_team)[0]
    bowling_team = team_encoder.transform(bowling_team)[0]
    wickets = len(batsmen)-2
    avg_strike_rate = get_avg_strike_rate(batsmen,batsmen_database_PP)
    avg_economy_rate = get_avg_economy_rate(bowlers,bowlers_database_PP)
    
    x_input = np.array([venue,innings,batting_team,bowling_team,avg_strike_rate
                       ,avg_economy_rate,wickets])
    x_input=x_input.astype(np.float32).reshape(1,-1)
    x_scaled = scaler.transform(x_input)
    y_pred = model.predict(x_scaled)
    prediction = round(y_pred[0])
    return prediction

