import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
import csv
from pathlib import Path
from datetime import datetime

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

class AdvancedFootballOnePredictor:
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
        home_id = self.base_predictor.data_fetcher.get_team_id(home_team, competition_id)
        away_id = self.base_predictor.data_fetcher.get_team_id(away_team, competition_id)

        
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
    
    def predict_with_random_forest(self, home_team: str, away_team: str, competition_id: int):
        """
        Predict match outcome using Random Forest model
        """
        # Train model
        model_result = self.train_random_forest_model(home_team, away_team)
        
        # Prepare current team data
        home_id = self.base_predictor.data_fetcher.get_team_id(home_team, competition_id)
        away_id = self.base_predictor.data_fetcher.get_team_id(away_team, competition_id)
        
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

class AdvancedFootballPredictor:
    def __init__(self, predictor):
        """
        Initialize with existing predictor to leverage its data fetching
        """
        self.base_predictor = predictor
        self.feature_importance = None
        self.best_model = None
        self.feature_columns = [
            'recent_goals_scored',
            'recent_goals_conceded',
            'form_points',
            'win_rate',
            'opponent_recent_goals_scored',
            'opponent_recent_goals_conceded',
            'opponent_form_points',
            'opponent_win_rate'
        ]
        
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
            
    
              
            # Modified label encoding: Convert [-1, 0, 1] to [0, 1, 2]
            next_winner = next_match.get('score', {}).get('winner')
            if next_winner == 'DRAW':
                label = 1  # Draw (was 0)
            elif next_winner == 'HOME_TEAM':
                label = 2 if is_home else 0  # Home win (was 1) or Away loss (was -1)
            elif next_winner == 'AWAY_TEAM':
                label = 0 if is_home else 2  # Away win (was -1) or Home loss (was 1)
            
            features.append(list(feature.values()))
            labels.append(label)
        
        return pd.DataFrame(features, columns=self.feature_columns), pd.Series(labels)
    
    def _calculate_recent_form(self, matches: List[Dict], team_name: str) -> Dict:
        """Calculate recent team form with additional metrics"""
        form_data = {
            'results': [],
            'goals_scored': [],
            'goals_conceded': [],
        }
        
        for match in matches[:10]:  # Consider last 10 matches
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

        # Calculate additional metrics
        clean_sheets = sum(1 for goals in form_data['goals_conceded'] if goals == 0)
        
        # Calculate scoring streak
        scoring_streak = 0
        for goals in form_data['goals_scored']:
            if goals > 0:
                scoring_streak += 1
            else:
                break

        # Calculate form points
        points_map = {'W': 3, 'D': 1, 'L': 0}
        form_points = sum(points_map.get(result, 0) for result in form_data['results'])
        
        return {
            'avg_goals_scored': np.mean(form_data['goals_scored']) if form_data['goals_scored'] else 0,
            'avg_goals_conceded': np.mean(form_data['goals_conceded']) if form_data['goals_conceded'] else 0,
            'form_points': form_points,
            'win_rate': form_data['results'].count('W') / len(form_data['results']) * 100 if form_data['results'] else 0,
            'clean_sheets': clean_sheets,
            'scoring_streak': scoring_streak
        }
    
    def analyze_feature_importance(self, X, model):
        """Analyze and store feature importance"""
        importance = model.feature_importances_
        features = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = features
        return self.feature_columns

    def train_ensemble_model(self, home_team: str, away_team: str, competition_id: int):
        """Train an ensemble of models with advanced techniques"""
        # Fetch and prepare data
        home_id = self.base_predictor.data_fetcher.get_team_id(home_team, competition_id)
        away_id = self.base_predictor.data_fetcher.get_team_id(away_team, competition_id)
        
        home_matches = self.base_predictor.data_fetcher.get_team_matches(home_id, days=365)
        away_matches = self.base_predictor.data_fetcher.get_team_matches(away_id, days=365)
        
        # Create datasets
        home_features, home_labels = self.create_match_dataset(home_matches, home_team)
        away_features, away_labels = self.create_match_dataset(away_matches, away_team)
        
        # Combine data
        X = pd.concat([home_features, away_features], ignore_index=True)
        y = pd.concat([home_labels, away_labels], ignore_index=True)
        
        # Convert to numpy arrays before scaling to remove feature names
        X_numpy = X.to_numpy()
        
        # Fit and transform with scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numpy)
        
        # Train models using numpy arrays (without feature names)
        models = {
            'rf': RandomForestClassifier(random_state=42),
            'gb': GradientBoostingClassifier(random_state=42),
            'xgb': xgb.XGBClassifier(random_state=42)
        }
        
        # Define parameter grids for each model
        param_grids = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced']
            },
            'gb': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            },
            'xgb': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'min_child_weight': [1, 3]
            }
        }
        
        # Train and evaluate models
        best_models = {}
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                model,
                param_grids[name],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_scaled, y)
            
            # Store best model version
            best_models[name] = grid_search.best_estimator_
            
            # Cross-validation score
            cv_score = cross_val_score(
                grid_search.best_estimator_,
                X_scaled,
                y,
                cv=5,
                scoring='accuracy'
            ).mean()
            
            if cv_score > best_score:
                best_score = cv_score
                best_model_name = name
                self.best_model = grid_search.best_estimator_
        
        # Analyze feature importance for best model
        important_features = self.analyze_feature_importance(X_scaled, self.best_model)
        
        # Final model using only important features
        X_important = X_scaled[important_features]
        final_model = best_models[best_model_name]
        final_model.fit(X_important, y)
        
        # Save both the feature columns and the numpy-trained model
        model_data = {
            'model': final_model,
            'scaler': scaler,
            'feature_columns': self.feature_columns,
            'accuracy': best_score
        }
        
        joblib.dump(model_data, 'football_prediction_model.joblib')
        return model_data

    def predict_with_ensemble(self, home_team: str, away_team: str, competition_id: int):
        """Make predictions using the ensemble model"""
        try:
            # Load or train model
            try:
                model_data = joblib.load('football_prediction_model.joblib')
            except:
                model_data = self.train_ensemble_model(home_team, away_team, competition_id)
            
            # Prepare current match features
            home_id = self.base_predictor.data_fetcher.get_team_id(home_team, competition_id)
            away_id = self.base_predictor.data_fetcher.get_team_id(away_team, competition_id)
            
            home_matches = self.base_predictor.data_fetcher.get_team_matches(home_id, days=120)
            away_matches = self.base_predictor.data_fetcher.get_team_matches(away_id, days=120)
            
            home_form = self._calculate_recent_form(home_matches[:10], home_team)
            away_form = self._calculate_recent_form(away_matches[:10], away_team)
            
            # Create feature vector using only the defined features
            features = pd.DataFrame([{
                'recent_goals_scored': home_form['avg_goals_scored'],
                'recent_goals_conceded': home_form['avg_goals_conceded'],
                'form_points': home_form['form_points'],
                'win_rate': home_form['win_rate'],
                'opponent_recent_goals_scored': away_form['avg_goals_scored'],
                'opponent_recent_goals_conceded': away_form['avg_goals_conceded'],
                'opponent_form_points': away_form['form_points'],
                'opponent_win_rate': away_form['win_rate']
            }], columns=self.feature_columns)
            
            # Convert features to numpy array before scaling
            features_numpy = features.to_numpy()
            
            # Scale features
            features_scaled = model_data['scaler'].transform(features_numpy)
            
            # Make prediction using numpy array
            prediction = model_data['model'].predict(features_scaled)[0]
            probabilities = model_data['model'].predict_proba(features_scaled)[0]
            
            # Interpret prediction
            outcomes = ['Away Win', 'Draw', 'Home Win']
            predicted_outcome = outcomes[prediction]
            
            return {
                'predicted_outcome': predicted_outcome,
                'home_win_prob': round(probabilities[2] * 100, 1),
                'away_win_prob': round(probabilities[0] * 100, 1),
                'draw_prob': round(probabilities[1] * 100, 1),
                'model_accuracy': round(model_data['accuracy'] * 100, 1),
                'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise

class FootballDataFetcher:
    """Handles all API interactions to fetch football data from football-data.org"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': api_key}
        self._team_id_cache = {}
        self._team_matches_cache = {}
        self._competition_cache = {}
        self._initialize_competitions()

    def _initialize_competitions(self):
        """Fetch and cache all available competitions"""
        try:
            url = f"{self.base_url}/competitions"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                competitions = response.json().get('competitions', [])
                for comp in competitions:
                    comp_id = comp.get('id')
                    if comp_id:
                        self._competition_cache[comp_id] = {
                            'name': comp.get('name'),
                            'type': comp.get('type'),
                            'country': comp.get('area', {}).get('name')
                        }
            else:
                logging.error(f"Failed to fetch competitions: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error initializing competitions: {str(e)}")

    def get_team_id(self, team_short_name: str, competition_id: int) -> int:
        """Get team ID using short name and competition ID"""
        # Check cache first
        cache_key = f"{team_short_name}_{competition_id}"
        if cache_key in self._team_id_cache:
            return self._team_id_cache[cache_key]

        try:
            url = f"{self.base_url}/competitions/{competition_id}/teams"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                teams = response.json().get('teams', [])
                
                for team in teams:
                    if team.get('shortName') == team_short_name:
                        self._team_id_cache[cache_key] = team['id']
                        return team['id']
                    
            elif response.status_code == 429:
                print("Rate limit reached. Waiting 60 seconds...")
                time.sleep(60)
                return self.get_team_id(team_short_name, competition_id)
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching teams: {str(e)}")
            time.sleep(5)
            raise
            
        raise ValueError(f"Team not found: {team_short_name} in competition {competition_id}")

    def get_team_matches(self, team_id: int, days: int = 90) -> List[Dict]:
        """Get recent matches from all competitions"""
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
            'limit': 100  # Increased limit to get more matches
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                matches = response.json().get('matches', [])
                
                # Enrich match data with competition info
                for match in matches:
                    comp_id = match.get('competition', {}).get('id')
                    if comp_id in self._competition_cache:
                        match['competition_info'] = self._competition_cache[comp_id]
                
                sorted_matches = sorted(matches, key=lambda x: x.get('utcDate', ''), reverse=True)
                
                # Cache the result
                self._team_matches_cache[cache_key] = sorted_matches
                return sorted_matches
            elif response.status_code == 429:
                print("Rate limit reached. Waiting 60 seconds...")
                time.sleep(60)
                return self.get_team_matches(team_id, days)
            else:
                raise Exception(f"API request failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching matches: {str(e)}")
            time.sleep(5)
            raise Exception(f"Error fetching matches: {str(e)}")

    def get_available_competitions(self) -> Dict:
        """Return list of available competitions"""
        return self._competition_cache

class FootballPredictor:
    def __init__(self, api_key: str):
        self.data_fetcher = FootballDataFetcher(api_key)
        self.logger = logging.getLogger(__name__)
        
    def predict_match(self, home_team_short: str, away_team_short: str, competition_id: int) -> Dict:
        """Make match prediction for teams in a specific competition"""
        try:
            home_id = self.data_fetcher.get_team_id(home_team_short, competition_id)
            away_id = self.data_fetcher.get_team_id(away_team_short, competition_id)
            
            home_matches = self.data_fetcher.get_team_matches(home_id, days=90)
            away_matches = self.data_fetcher.get_team_matches(away_id, days=90)
            
            home_form = self._calculate_team_form(home_matches, home_team_short)
            away_form = self._calculate_team_form(away_matches, away_team_short)
            
            # Enhanced prediction calculation
            home_points = home_form['form_points']
            away_points = away_form['form_points']
            max_possible_points = 30
            
            home_form_factor = (home_points / max_possible_points) * 100
            away_form_factor = (away_points / max_possible_points) * 100
            
            # Consider goal scoring and conceding records
            home_goal_factor = home_form['avg_goals_scored'] - home_form['avg_goals_conceded']
            away_goal_factor = away_form['avg_goals_scored'] - away_form['avg_goals_conceded']
            
            home_advantage = 7
            
            home_win_prob = (home_form_factor + home_advantage + (home_goal_factor * 5)) * 0.6
            away_win_prob = (away_form_factor + (away_goal_factor * 5)) * 0.5
            
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
                        'form_points': home_form['form_points'],
                        'avg_goals_scored': round(home_form['avg_goals_scored'], 2),
                        'avg_goals_conceded': round(home_form['avg_goals_conceded'], 2)
                    },
                    'away': {
                        'recent_form': away_form['form_trend'],
                        'form_points': away_form['form_points'],
                        'avg_goals_scored': round(away_form['avg_goals_scored'], 2),
                        'avg_goals_conceded': round(away_form['avg_goals_conceded'], 2)
                    }
                }
            }
                
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise

    def _calculate_team_form(self, matches: List[Dict], team_name: str) -> Dict:
        """Calculate team form based on recent matches from all competitions"""
        form_data = {
            'results': [],
            'goals_scored': [],
            'goals_conceded': [],
        }
        
        # Use all matches but prioritize domestic league matches
        sorted_matches = sorted(matches, 
                              key=lambda x: (x.get('competition', {}).get('type') == 'LEAGUE', x['utcDate']),
                              reverse=True)
        
        for match in sorted_matches[:10]:  # Consider last 10 matches
            winner = match.get('score', {}).get('winner')
            is_home = match['homeTeam']['shortName'] == team_name
            print(match['homeTeam']['shortName'])
            
            if winner == 'DRAW':
                form_data['results'].append('D')
            elif winner == 'HOME_TEAM':
                form_data['results'].append('W' if is_home else 'L')
            elif winner == 'AWAY_TEAM':
                form_data['results'].append('W' if not is_home else 'L')
            
            score = match.get('score', {}).get('fullTime', {})
            if is_home:
                goals_scored = score.get('home', 0)
                goals_conceded = score.get('away', 0)
            else:
                goals_scored = score.get('away', 0)
                goals_conceded = score.get('home', 0)
                
            form_data['goals_scored'].append(goals_scored)
            form_data['goals_conceded'].append(goals_conceded)
        
        points_map = {'W': 3, 'D': 1, 'L': 0}
        form_points = sum(points_map[result] for result in form_data['results'])
        
        return {
            'form_trend': ''.join(form_data['results']),
            'avg_goals_scored': np.mean(form_data['goals_scored']) if form_data['goals_scored'] else 0,
            'avg_goals_conceded': np.mean(form_data['goals_conceded']) if form_data['goals_conceded'] else 0,
            'form_points': form_points
        }


def format_predictions(home_team, away_team, original_pred, rf_pred, rf_pred_one, save_to_csv=True):
    formatted_output = f"""
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

Random Forest One Prediction:
- Predicted Outcome:    {rf_pred_one['predicted_outcome']}
- Home Win Probability: {rf_pred_one['home_win_prob']}%
- Draw Probability:     {rf_pred_one['draw_prob']}%
- Away Win Probability: {rf_pred_one['away_win_prob']}%
- Model Accuracy:       {rf_pred_one['model_accuracy']}%
"""
    
    if save_to_csv:
        # Prepare data for CSV
        prediction_data = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'home_team': home_team,
            'away_team': away_team,
            'original_home_win_prob': original_pred['home_win_prob'],
            'original_draw_prob': original_pred['draw_prob'],
            'original_away_win_prob': original_pred['away_win_prob'],
            'home_form': original_pred['team_form']['home']['recent_form'],
            'away_form': original_pred['team_form']['away']['recent_form'],
            'rf_predicted_outcome': rf_pred['predicted_outcome'],
            'rf_home_win_prob': rf_pred['home_win_prob'],
            'rf_draw_prob': rf_pred['draw_prob'],
            'rf_away_win_prob': rf_pred['away_win_prob'],
            'model_accuracy': rf_pred['model_accuracy'],
            'rf_pred_one_predicted_outcome': rf_pred_one['predicted_outcome'],
            'rf_one_home_win_prob': rf_pred_one['home_win_prob'],
            'rf_one_draw_prob': rf_pred_one['draw_prob'],
            'rf_one_away_win_prob': rf_pred_one['away_win_prob'],
            'rf_one_model_accuracy': rf_pred_one['model_accuracy']
        }
        
        # Create predictions directory if it doesn't exist
        Path('predictions').mkdir(exist_ok=True)
        
        # Define CSV file path
        csv_file = 'predictions/match_predictions.csv'
        file_exists = Path(csv_file).exists()
        
        # Write to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=prediction_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(prediction_data)
    
    return formatted_output

# Example usage
if __name__ == "__main__":
    API_KEY = "75346525f8f34616b629ad0c4fe9e471"
    predictor = FootballPredictor(API_KEY)
    advanced_predictor = AdvancedFootballPredictor(predictor)
    advanced_predictor_one = AdvancedFootballOnePredictor(predictor)
    
    try:
        # Example using Bundesliga (ID: 2002)
        home_team_short = "Getafe"
        away_team_short = "Valladolid"
        competition_id = 2014
        
        # Original method prediction
        original_prediction = predictor.predict_match(home_team_short, away_team_short, competition_id)
        # Random Forest prediction
        rf_prediction_one = advanced_predictor_one.predict_with_random_forest(home_team_short, away_team_short, competition_id)
        # Random Forest prediction
        rf_prediction = advanced_predictor.predict_with_ensemble(home_team_short, away_team_short, competition_id)
        
        print(format_predictions(home_team_short, away_team_short, original_prediction, rf_prediction, rf_prediction_one))
        
    except Exception as e:
        print(f"Error: {str(e)}")