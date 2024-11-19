import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import time

class AdvancedFootballPredictor:
    def __init__(self, predictor):
        """
        Initialize with existing predictor to leverage its data fetching
        """
        self.base_predictor = predictor
        
    def create_match_dataset(self, matches: List[Dict], team_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform match data into a structured dataset for machine learning
        """
        features = []
        labels = []
        
        # Process league matches
        league_matches = [
            match for match in matches 
            if match.get('competition', {}).get('type') == 'LEAGUE'
            and match.get('status') == 'FINISHED'
        ]
        league_matches.sort(key=lambda x: x['utcDate'], reverse=True)
        
        for i in range(len(league_matches) - 1):
            current_match = league_matches[i]
            next_match = league_matches[i+1]
            
            is_home = current_match['homeTeam']['shortName'] == team_name
            opponent_name = current_match['awayTeam']['shortName'] if is_home else current_match['homeTeam']['shortName']
            
            # Extract match features
           
            
            # Calculate recent form
            recent_matches = league_matches[i:i+10]
            form_data = self._calculate_recent_form(recent_matches, team_name)
            opponent_form_data = self._calculate_recent_form(recent_matches, opponent_name)

            feature = {
            'recent_goals_scored': form_data['avg_goals_scored'],
            'recent_goals_conceded': form_data['avg_goals_conceded'],
            'form_points': form_data['form_points'],
            'win_rate': form_data['win_rate'],
            'opponent_recent_goals_scored': opponent_form_data['avg_goals_scored'],
            'opponent_recent_goals_conceded': opponent_form_data['avg_goals_conceded'],
            'opponent_form_points': opponent_form_data['form_points'],
            'opponent_win_rate': opponent_form_data['win_rate']
        }
            
    
              
            # Determine match outcome for next match
            next_winner = next_match.get('score', {}).get('winner')
            if next_winner == 'DRAW':
                label = 0  # Draw
            elif next_winner == 'HOME_TEAM':
                label = 1 if is_home else -1  # Home win or Away loss
            elif next_winner == 'AWAY_TEAM':
                label = -1 if is_home else 1  # Away win or Home loss
            
            features.append(list(feature.values()))
            labels.append(label)
        
        return pd.DataFrame(features), pd.Series(labels)
    
    def _calculate_recent_form(self, matches: List[Dict], team_name: str) -> Dict:
        """
        Calculate recent team form
        """
        form_data = {
            'results': [],
            'goals_scored': [],
            'goals_conceded': [],
        }
        
        for match in matches:
            is_home = match['homeTeam']['shortName'] == team_name
            winner = match.get('score', {}).get('winner')
            
            # Determine result
            if winner == 'DRAW':
                form_data['results'].append('D')
            elif winner == 'HOME_TEAM':
                form_data['results'].append('W' if is_home else 'L')
            elif winner == 'AWAY_TEAM':
                form_data['results'].append('W' if not is_home else 'L')
            
            # Track goals
            score = match.get('score', {}).get('fullTime', {})
            if is_home:
                goals_scored = score.get('home', 0)
                goals_conceded = score.get('away', 0)
            else:
                goals_scored = score.get('away', 0)
                goals_conceded = score.get('home', 0)
            
            form_data['goals_scored'].append(goals_scored)
            form_data['goals_conceded'].append(goals_conceded)
        
        # Calculate form points
        points_map = {'W': 3, 'D': 1, 'L': 0}
        form_points = sum(points_map.get(result, 0) for result in form_data['results'])
        
        return {
            'avg_goals_scored': np.mean(form_data['goals_scored']) if form_data['goals_scored'] else 0,
            'avg_goals_conceded': np.mean(form_data['goals_conceded']) if form_data['goals_conceded'] else 0,
            'form_points': form_points,
            'win_rate': form_data['results'].count('W') / len(form_data['results']) * 100 if form_data['results'] else 0
        }
    
    def train_random_forest_model(self, home_team: str, away_team: str):
        """
        Train a Random Forest model for match prediction
        """
        # Single API call for team IDs
        home_id = self.base_predictor.data_fetcher.get_team_id(home_team)
        away_id = self.base_predictor.data_fetcher.get_team_id(away_team)
        
        # Single API call for matches
        home_matches = self.base_predictor.data_fetcher.get_team_matches(home_id, days=365)
        away_matches = self.base_predictor.data_fetcher.get_team_matches(away_id, days=365)
        
        # Create datasets
        home_features, home_labels = self.create_match_dataset(home_matches, home_team)
        away_features, away_labels = self.create_match_dataset(away_matches, away_team)
        
        # Combine and prepare data
        X = pd.concat([home_features, away_features], ignore_index=True)
        y = pd.concat([home_labels, away_labels], ignore_index=True)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        return {
            'model': rf_model,
            'scaler': scaler,
            'accuracy': rf_model.score(X_test, y_test)
        }
    
    def predict_with_random_forest(self, home_team: str, away_team: str):
        """
        Predict match outcome using Random Forest model
        """
        # Train model
        model_result = self.train_random_forest_model(home_team, away_team)
        
        # Prepare current team data
        home_id = self.base_predictor.data_fetcher.get_team_id(home_team)
        away_id = self.base_predictor.data_fetcher.get_team_id(away_team)
        
        home_matches = self.base_predictor.data_fetcher.get_team_matches(home_id, days=120)
        away_matches = self.base_predictor.data_fetcher.get_team_matches(away_id, days=120)
        
        # Calculate recent form
        home_form = self._calculate_recent_form(home_matches[:10], home_team)
        away_form = self._calculate_recent_form(away_matches[:10], away_team)
        
        # Prepare input features
        input_features = [
        home_form['avg_goals_scored'],
        home_form['avg_goals_conceded'],
        home_form['form_points'],
        home_form['win_rate'],
        away_form['avg_goals_scored'],
        away_form['avg_goals_conceded'],
        away_form['form_points'],
        away_form['win_rate']
        ]
        
        # Scale features
        input_scaled = model_result['scaler'].transform([input_features])
        
        # Predict
        prediction = model_result['model'].predict(input_scaled)[0]
        probabilities = model_result['model'].predict_proba(input_scaled)[0]
        
        
        # Interpret prediction
        if prediction == 0:
            outcome = 'Draw'
            home_win_prob = probabilities[1]
            away_win_prob = probabilities[2]
            draw_prob = probabilities[0]
        elif prediction == 1:
            outcome = 'Home Win'
            home_win_prob = probabilities[1]
            away_win_prob = probabilities[2]
            draw_prob = probabilities[0]
        else:
            outcome = 'Away Win'
            home_win_prob = probabilities[1]
            away_win_prob = probabilities[2]
            draw_prob = probabilities[0]
        
        return {
            'predicted_outcome': outcome,
            'home_win_prob': round(home_win_prob * 100, 1),
            'away_win_prob': round(away_win_prob * 100, 1),
            'draw_prob': round(draw_prob * 100, 1),
            'model_accuracy': round(model_result['accuracy'] * 100, 1)
        }

class FootballDataFetcher:
    """Handles all API interactions to fetch football data from football-data.org"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': api_key}
        self.competitions = {
            'PL': 2021,    # Premier League
            'PD': 2014,    # La Liga
            'BL1': 2002,   # Bundesliga
            'SA': 2019,    # Serie A
            'FL1': 2015,   # Ligue 1
            'CL': 2001     # UEFA Champions League
        }
        self._team_id_cache = {}
        self._team_matches_cache = {}

    def get_team_id(self, team_name: str) -> int:
        """Get team ID with caching and improved name matching"""
        # Check cache first
        if team_name in self._team_id_cache:
            return self._team_id_cache[team_name]

        for competition_id in self.competitions.values():
            try:
                url = f"{self.base_url}/competitions/{competition_id}/teams"
                response = requests.get(url, headers=self.headers)
                
                if response.status_code == 200:
                    teams = response.json().get('teams', [])
                    
                    # Normalize the search team name
                    search_name = team_name.lower().strip()
                    
                    for team in teams:
                        # Check multiple name variations
                        team_variations = [
                            team.get('name', '').lower(),
                            team.get('shortName', '').lower(),
                            team.get('tla', '').lower()
                        ]
                        
                        # Special case handling for common variations
                        if 'united' in search_name:
                            team_variations.append(team.get('name', '').lower().replace('united', 'utd'))
                        
                        # Check if search name matches any variation
                        if any(search_name in variation or variation in search_name 
                              for variation in team_variations if variation):
                            # Cache the result
                            self._team_id_cache[team_name] = team['id']
                            return team['id']
                
                elif response.status_code == 429:
                    # Rate limit handling
                    print("Rate limit reached. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                    
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching teams for competition {competition_id}: {str(e)}")
                time.sleep(5)  # Wait before retrying
                continue
                
        raise ValueError(f"Team not found: {team_name}. Please check the team name and try again.")
        
    def get_team_matches(self, team_id: int, days: int = 90) -> List[Dict]:
        """Get recent matches for a team with caching and rate limit handling"""
        # Check cache first
        cache_key = (team_id, days)
        if cache_key in self._team_matches_cache:
            return self._team_matches_cache[cache_key]

        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        url = f"{self.base_url}/teams/{team_id}/matches"
        params = {
            'dateFrom': from_date,
            'dateTo': to_date,
            'status': 'FINISHED',
            'competitions': ','.join([str(comp_id) for comp_id in [2021, 2016]]),  # Premier League and Championship IDs
            'limit': 20
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                matches = response.json().get('matches', [])
                sorted_matches = sorted(matches, key=lambda x: x.get('utcDate', ''), reverse=True)
                
                # Cache the result
                self._team_matches_cache[cache_key] = sorted_matches
                return sorted_matches
            elif response.status_code == 429:
                # Rate limit handling
                print("Rate limit reached. Waiting 60 seconds...")
                time.sleep(60)
                return self.get_team_matches(team_id, days)
            else:
                raise Exception(f"API request failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching matches: {str(e)}")
            time.sleep(5)  # Wait before retrying
            raise Exception(f"Error fetching matches: {str(e)}")

class FootballPredictor:
    def __init__(self, api_key: str):
        self.data_fetcher = FootballDataFetcher(api_key)
        self.logger = logging.getLogger(__name__)
        
    def predict_match(self, home_team: str, away_team: str) -> Dict:
        """Make match prediction with reduced API calls"""
        try:
            # Single API call for team IDs
            home_id = self.data_fetcher.get_team_id(home_team)
            away_id = self.data_fetcher.get_team_id(away_team)
            
            # Single API call for matches for both teams
            home_matches = self.data_fetcher.get_team_matches(home_id, days=90)
            away_matches = self.data_fetcher.get_team_matches(away_id, days=90)
            
            # Rest of the prediction logic remains the same
            home_form = self._calculate_team_form(home_matches, home_team)
            away_form = self._calculate_team_form(away_matches, away_team)
            
            # Prediction calculation logic
            home_points = home_form['form_points']
            away_points = away_form['form_points']
            max_possible_points = 30
            
            home_form_factor = (home_points / max_possible_points) * 100
            away_form_factor = (away_points / max_possible_points) * 100
            
            home_advantage = 7
            
            home_win_prob = (home_form_factor + home_advantage) * 0.6
            away_win_prob = away_form_factor * 0.5
            
            form_difference = abs(home_points - away_points)
            draw_prob = max(20, 30 - form_difference * 2)
            
            total = home_win_prob + away_win_prob + draw_prob
            home_win_prob = (home_win_prob / total) * 100
            away_win_prob = (away_win_prob / total) * 100
            draw_prob = (draw_prob / total) * 100
            
            return {
                'home_win_prob': round(home_win_prob, 1),
                'draw_prob': round(draw_prob, 1),
                'away_win_prob': round(away_win_prob, 1),
                'team_form': {
                    'home': {
                        'recent_form': home_form['form_trend'],
                        'form_points': home_form['form_points']
                    },
                    'away': {
                        'recent_form': away_form['form_trend'],
                        'form_points': away_form['form_points']
                    }
                }
            }
                
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise

    def _calculate_team_form(self, matches: List[Dict], team_name: str) -> Dict:
        """Calculate team form based on 5 most recent LEAGUE matches"""
        form_data = {
            'results': [],
            'goals_scored': [],
            'goals_conceded': [],
        }
        
        # Process only Premier League matches and sort by date
        league_matches = [
            match for match in matches 
            if match.get('competition', {}).get('type') == 'LEAGUE'
            and match.get('status') == 'FINISHED'
        ]
        
        # Sort matches by date in descending order (most recent first)
        league_matches.sort(key=lambda x: x['utcDate'], reverse=True)
        
        # Take only the 5 most recent league matches
        for match in league_matches[:10]:
            winner = match.get('score', {}).get('winner')
            is_home = match['homeTeam']['shortName'] == team_name
            
            # Determine result based on winner and team position
            if winner == 'DRAW':
                form_data['results'].append('D')
            elif winner == 'HOME_TEAM':
                form_data['results'].append('W' if is_home else 'L')
            elif winner == 'AWAY_TEAM':
                form_data['results'].append('W' if not is_home else 'L')
            
            # Track goals
            score = match.get('score', {}).get('fullTime', {})
            if is_home:
                goals_scored = score.get('home', 0)
                goals_conceded = score.get('away', 0)
            else:
                goals_scored = score.get('away', 0)
                goals_conceded = score.get('home', 0)
                
            form_data['goals_scored'].append(goals_scored)
            form_data['goals_conceded'].append(goals_conceded)
        
        # Calculate form points
        points_map = {'W': 3, 'D': 1, 'L': 0}
        form_points = sum(points_map[result] for result in form_data['results'])
        
        return {
            'form_trend': ''.join(form_data['results']),
            'avg_goals_scored': np.mean(form_data['goals_scored']) if form_data['goals_scored'] else 0,
            'avg_goals_conceded': np.mean(form_data['goals_conceded']) if form_data['goals_conceded'] else 0,
            'form_points': form_points
        }

def format_predictions(home_team, away_team, original_pred, rf_pred):
   return f"""
Match Prediction: {home_team} vs {away_team}
---------------------------------------
Original Prediction:
- Home Win Probability: {original_pred['home_win_prob']}%
- Draw Probability:     {original_pred['draw_prob']}%
- Away Win Probability: {original_pred['away_win_prob']}%

Team Form:
- {home_team}:
 * Recent Form: {original_pred['team_form']['home']['recent_form']}
 * Form Points: {original_pred['team_form']['home']['form_points']}

- {away_team}:
 * Recent Form: {original_pred['team_form']['away']['recent_form']}
 * Form Points: {original_pred['team_form']['away']['form_points']}

Random Forest Prediction:
- Predicted Outcome:    {rf_pred['predicted_outcome']}
- Home Win Probability: {rf_pred['home_win_prob']}%
- Draw Probability:     {rf_pred['draw_prob']}%
- Away Win Probability: {rf_pred['away_win_prob']}%
- Model Accuracy:       {rf_pred['model_accuracy']}%
"""

# Example usage
if __name__ == "__main__":
    API_KEY = "75346525f8f34616b629ad0c4fe9e471"
    predictor = FootballPredictor(API_KEY)
    advanced_predictor = AdvancedFootballPredictor(predictor)
    
    try:
        # Let's debug the match fetching
        home_team = "Leicester City"
        away_team = "Chelsea"
       # Original method prediction
        original_prediction = predictor.predict_match(home_team, away_team)
        
        # Random Forest prediction
        rf_prediction = advanced_predictor.predict_with_random_forest(home_team, away_team)
        
        print(format_predictions(home_team, away_team, original_prediction, rf_prediction))
        
    except Exception as e:
        print(f"Error: {str(e)}")