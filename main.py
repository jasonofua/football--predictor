import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
import requests
import warnings
from datetime import datetime, timedelta
from tabulate import tabulate
from colorama import Fore, Style, init
import json
from typing import Dict, List, Tuple
import logging

# Initialize colorama
init()

class FootballPredictor:
    def __init__(self, api_key: str):
        """Initialize the football predictor with API key and models"""
        self.api_key = api_key
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)

    def fit(self, training_data: pd.DataFrame):
        """
        Fit the model and scaler with historical match data
        
        Parameters:
        - training_data: DataFrame containing historical match data
        """
        # Prepare features for each match in training data
        feature_list = []
        target_list = []
        
        for idx in range(len(training_data)):
            match_data = training_data.iloc[idx:idx+1]
            features = self.prepare_advanced_features(match_data)
            feature_list.append(features)
            
            # Generate target (dummy for now - you would use actual match results)
            target_list.append(np.random.choice([0, 1]))
        
        # Combine all features and targets
        X = pd.concat(feature_list, ignore_index=True)
        y = np.array(target_list)

        # Fit the scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Fit the model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def get_recent_matches(self, home_team: str, away_team: str, 
                          days: int = 90) -> pd.DataFrame:
        """Get recent match data for both teams"""
        start_date = datetime.now() - timedelta(days=days)
        
        dummy_data = {
            'date': pd.date_range(start=start_date, periods=10),
            'home_team': [home_team] * 5 + [away_team] * 5,
            'away_team': [away_team] * 5 + [home_team] * 5,
            'shots_data': [pd.DataFrame({
                'distance': np.random.normal(15, 5, 10),
                'angle': np.random.normal(45, 15, 10),
                'situation': np.random.choice(['open_play', 'set_piece', 'penalty'], 10),
                'on_target': np.random.choice([True, False], 10),
                'result': np.random.choice(['goal', 'miss', 'save'], 10)
            }) for _ in range(10)],
            'possession_data': [pd.DataFrame({
                'total_possession_time': np.random.normal(90, 5, 1),
                'time_in_opp_third': np.random.normal(30, 5, 1)
            }) for _ in range(10)],
            'tackles': np.random.normal(20, 5, 10),
            'interceptions': np.random.normal(15, 3, 10),
            'fouls': np.random.normal(12, 3, 10),
            'opponent_passes': np.random.normal(500, 50, 10)
        }
        
        return pd.DataFrame(dummy_data)

    def prepare_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare basic features for prediction"""
        features = pd.DataFrame()
        
        # Calculate basic stats for each team
        for team in [data['home_team'].iloc[0], data['away_team'].iloc[0]]:
            team_matches = data[
                (data['home_team'] == team) | 
                (data['away_team'] == team)
            ]
            
            team_features = {
                f'{team}_tackles_avg': team_matches['tackles'].mean(),
                f'{team}_interceptions_avg': team_matches['interceptions'].mean(),
                f'{team}_fouls_avg': team_matches['fouls'].mean(),
                f'{team}_passes_against_avg': team_matches['opponent_passes'].mean()
            }
            
            for key, value in team_features.items():
                features[key] = [value]
        
        return features

    def predict_match(self, home_team: str, away_team: str, 
                     match_date: datetime) -> Dict:
        """Make basic match prediction"""
        if not self.is_fitted:
            self.logger.warning("Models not fitted. Training with dummy data...")
            training_data = self.get_recent_matches(home_team, away_team, days=180)
            self.fit(training_data)

        # Get recent match data
        recent_matches = self.get_recent_matches(home_team, away_team)
        
        # Prepare features
        features = self.prepare_advanced_features(recent_matches.iloc[0:1])
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        win_prob = self.model.predict_proba(features_scaled)[0]
        
        return {
            'home_win_prob': win_prob[1],
            'draw_prob': 0.25,  # Dummy value
            'away_win_prob': win_prob[0],
            'predicted_score': {'home': 2, 'away': 1}  # Dummy values
        }

    def display_detailed_prediction(self, home_team: str, away_team: str,
                                  prediction: Dict, match_date: datetime) -> None:
        """Display detailed match prediction"""
        print(f"\nMatch Prediction: {home_team} vs {away_team}")
        print(f"Date: {match_date.strftime('%Y-%m-%d')}\n")
        
        print("Win Probabilities:")
        print(f"{home_team}: {prediction['home_win_prob']:.1%}")
        print(f"Draw: {prediction['draw_prob']:.1%}")
        print(f"{away_team}: {prediction['away_win_prob']:.1%}\n")
        
        print("Predicted Score:")
        print(f"{home_team} {prediction['predicted_score']['home']} - "
              f"{prediction['predicted_score']['away']} {away_team}")

class AdvancedMetricsCalculator:
    """Calculate advanced football metrics"""
    
    @staticmethod
    def calculate_xG(shots_data: pd.DataFrame) -> float:
        """
        Calculate Expected Goals (xG) based on shot quality
        
        Parameters:
        - shot_distance: Distance from goal
        - shot_angle: Angle of the shot
        - situation: Type of play (open play, set piece, penalty)
        """
        def shot_probability(distance, angle, situation):
            # Base probability based on distance
            base_prob = 1 / (1 + np.exp(0.1 * distance - 1))
            
            # Angle modifier
            angle_mod = np.sin(np.radians(angle))
            
            # Situation modifier
            situation_mods = {
                'penalty': 0.76,
                'set_piece': 0.08,
                'open_play': 0.12,
                'counter': 0.15
            }
            
            return base_prob * angle_mod * situation_mods.get(situation, 0.1)
        
        xG = shots_data.apply(
            lambda x: shot_probability(
                x['distance'],
                x['angle'],
                x['situation']
            ),
            axis=1
        ).sum()
        
        return xG

    @staticmethod
    def calculate_ppda(team_data: pd.DataFrame) -> float:
        """
        Calculate Passes Allowed Per Defensive Action (PPDA)
        Lower PPDA = Higher pressing intensity
        """
        defensive_actions = (
            team_data['tackles'].sum() +
            team_data['interceptions'].sum() +
            team_data['fouls'].sum()
        )
        
        opponent_passes = team_data['opponent_passes'].sum()
        
        return opponent_passes / defensive_actions if defensive_actions > 0 else 0

    @staticmethod
    def calculate_field_tilt(possession_data: pd.DataFrame) -> float:
        """
        Calculate Field Tilt (percentage of possession in opposing third)
        """
        total_possession = possession_data['total_possession_time'].sum()
        opp_third_time = possession_data['time_in_opp_third'].sum()
        
        return (opp_third_time / total_possession * 100) if total_possession > 0 else 0

    @staticmethod
    def calculate_shot_quality(shots_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze shot quality metrics
        """
        total_shots = len(shots_data)
        if total_shots == 0:
            return {'shot_quality': 0, 'shot_conversion': 0}
            
        shots_on_target = len(shots_data[shots_data['on_target'] == True])
        goals = len(shots_data[shots_data['result'] == 'goal'])
        
        return {
            'shot_quality': shots_on_target / total_shots,
            'shot_conversion': goals / total_shots if total_shots > 0 else 0
        }
class EnhancedFootballPredictor(FootballPredictor):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.metrics_calculator = AdvancedMetricsCalculator()
        
        # Initialize additional models
        self.xg_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.shots_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.ppda_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.field_tilt_model = RandomForestRegressor(n_estimators=200, random_state=42)
        
        self.advanced_scalers = {
            'xg': StandardScaler(),
            'shots': StandardScaler(),
            'ppda': StandardScaler(),
            'field_tilt': StandardScaler()
        }
        
    def fit(self, training_data: pd.DataFrame):
        """Fit all models and scalers with historical match data"""
        super().fit(training_data)
        
        # Prepare features for advanced metrics
        X = self.prepare_advanced_features(training_data)
        
        # Prepare dummy targets for advanced metrics
        # In a real implementation, you would use actual historical data
        y_xg = np.random.normal(1.5, 0.5, len(training_data))
        y_shots = np.random.normal(12, 3, len(training_data))
        y_ppda = np.random.normal(10, 2, len(training_data))
        y_field_tilt = np.random.normal(50, 10, len(training_data))
        
        # Fit advanced metric models and scalers
        for model, scaler, y in [
            (self.xg_model, self.advanced_scalers['xg'], y_xg),
            (self.shots_model, self.advanced_scalers['shots'], y_shots),
            (self.ppda_model, self.advanced_scalers['ppda'], y_ppda),
            (self.field_tilt_model, self.advanced_scalers['field_tilt'], y_field_tilt)
        ]:
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)

    def predict_match(self, home_team: str, away_team: str, match_date: datetime) -> Dict:
        """Enhanced match predictions including advanced metrics"""
        if not self.is_fitted:
            self.logger.warning("Models not fitted. Training with dummy data...")
            training_data = self.get_recent_matches(home_team, away_team, days=180)
            self.fit(training_data)

        # Get basic predictions
        predictions = super().predict_match(home_team, away_team, match_date)
        
        # Get features for advanced predictions
        features = self.prepare_advanced_features(self.get_recent_matches(home_team, away_team))
        
        # Make advanced predictions
        advanced_predictions = {}
        for metric, model, scaler in [
            ('xG', self.xg_model, self.advanced_scalers['xg']),
            ('shots', self.shots_model, self.advanced_scalers['shots']),
            ('ppda', self.ppda_model, self.advanced_scalers['ppda']),
            ('field_tilt', self.field_tilt_model, self.advanced_scalers['field_tilt'])
        ]:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            
            if metric == 'xG':
                advanced_predictions['home_xG'] = prediction
                advanced_predictions['away_xG'] = prediction * 0.8  # Dummy adjustment for away team
            elif metric == 'shots':
                advanced_predictions['total_shots'] = prediction
            elif metric == 'ppda':
                advanced_predictions['home_ppda'] = prediction
                advanced_predictions['away_ppda'] = prediction * 1.2  # Dummy adjustment for away team
            elif metric == 'field_tilt':
                advanced_predictions['home_field_tilt'] = prediction
                advanced_predictions['away_field_tilt'] = 100 - prediction
        
        predictions.update(advanced_predictions)
        return predictions

if __name__ == "__main__":
    # Initialize predictor with API key
    API_KEY = "aac7116ce0e240ae92e90fc35e4beb4b"
    predictor = FootballPredictor(API_KEY)
    
    # Example prediction
    home_team = "AC Milan W"
    away_team = "Como W"
    match_date = datetime.now()

    historical_data = predictor.get_recent_matches(home_team, away_team, days=180)
    predictor.fit(historical_data)
    
    # Get and display prediction
    prediction = predictor.predict_match(home_team, away_team, match_date)
    predictor.display_detailed_prediction(home_team, away_team, prediction, match_date)





# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# import requests
# import warnings
# from datetime import datetime, timedelta
# from tabulate import tabulate
# from colorama import Fore, Style, init
# import json
# from typing import Dict, List, Tuple
# import logging

# # Initialize colorama
# init()


# class FootballPredictor:
#     def __init__(self, api_key: str):
#         """Initialize the football predictor with API key and models"""
#         self.api_key = api_key
#         self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#         self.scaler = StandardScaler()
#         self.logger = logging.getLogger(__name__)

#     def get_recent_matches(self, home_team: str, away_team: str, 
#                           days: int = 90) -> pd.DataFrame:
#         """
#         Get recent match data for both teams
        
#         Parameters:
#         - home_team: Name of home team
#         - away_team: Name of away team
#         - days: Number of days of history to fetch
#         """
#         start_date = datetime.now() - timedelta(days=days)
        
#         # In a real implementation, this would fetch data from an API
#         # For now, return dummy data structure
#         dummy_data = {
#             'date': pd.date_range(start=start_date, periods=10),
#             'home_team': [home_team] * 5 + [away_team] * 5,
#             'away_team': [away_team] * 5 + [home_team] * 5,
#             'shots_data': [pd.DataFrame({
#                 'distance': np.random.normal(15, 5, 10),
#                 'angle': np.random.normal(45, 15, 10),
#                 'situation': np.random.choice(['open_play', 'set_piece', 'penalty'], 10),
#                 'on_target': np.random.choice([True, False], 10),
#                 'result': np.random.choice(['goal', 'miss', 'save'], 10)
#             }) for _ in range(10)],
#             'possession_data': [pd.DataFrame({
#                 'total_possession_time': np.random.normal(90, 5, 1),
#                 'time_in_opp_third': np.random.normal(30, 5, 1)
#             }) for _ in range(10)],
#             'tackles': np.random.normal(20, 5, 10),
#             'interceptions': np.random.normal(15, 3, 10),
#             'fouls': np.random.normal(12, 3, 10),
#             'opponent_passes': np.random.normal(500, 50, 10)
#         }
        
#         return pd.DataFrame(dummy_data)

#     def prepare_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Prepare basic features for prediction"""
#         features = pd.DataFrame()
        
#         # Calculate basic stats for each team
#         for team in [data['home_team'].iloc[0], data['away_team'].iloc[0]]:
#             team_matches = data[
#                 (data['home_team'] == team) | 
#                 (data['away_team'] == team)
#             ]
            
#             # Basic stats
#             team_features = {
#                 f'{team}_tackles_avg': team_matches['tackles'].mean(),
#                 f'{team}_interceptions_avg': team_matches['interceptions'].mean(),
#                 f'{team}_fouls_avg': team_matches['fouls'].mean(),
#                 f'{team}_passes_against_avg': team_matches['opponent_passes'].mean()
#             }
            
#             # Add features to DataFrame
#             for key, value in team_features.items():
#                 features[key] = [value]
        
#         return features

#     def predict_match(self, home_team: str, away_team: str, 
#                      match_date: datetime) -> Dict:
#         """Make basic match prediction"""
#         # Get recent match data
#         recent_matches = self.get_recent_matches(home_team, away_team)
        
#         # Prepare features
#         features = self.prepare_advanced_features(recent_matches)
        
#         # In a real implementation, this would use trained models
#         # For now, return dummy predictions
#         return {
#             'home_win_prob': 0.45,
#             'draw_prob': 0.25,
#             'away_win_prob': 0.30,
#             'predicted_score': {'home': 2, 'away': 1}
#         }

#     def display_detailed_prediction(self, home_team: str, away_team: str,
#                                   prediction: Dict, match_date: datetime) -> None:
#         """Display basic match prediction"""
#         print(f"\nMatch Prediction: {home_team} vs {away_team}")
#         print(f"Date: {match_date.strftime('%Y-%m-%d')}\n")
        
#         print("Win Probabilities:")
#         print(f"{home_team}: {prediction['home_win_prob']:.1%}")
#         print(f"Draw: {prediction['draw_prob']:.1%}")
#         print(f"{away_team}: {prediction['away_win_prob']:.1%}\n")
        
#         print("Predicted Score:")
#         print(f"{home_team} {prediction['predicted_score']['home']} - "
#               f"{prediction['predicted_score']['away']} {away_team}")

# class AdvancedMetricsCalculator:
#     """Calculate advanced football metrics"""
    
#     @staticmethod
#     def calculate_xG(shots_data: pd.DataFrame) -> float:
#         """
#         Calculate Expected Goals (xG) based on shot quality
        
#         Parameters:
#         - shot_distance: Distance from goal
#         - shot_angle: Angle of the shot
#         - situation: Type of play (open play, set piece, penalty)
#         """
#         def shot_probability(distance, angle, situation):
#             # Base probability based on distance
#             base_prob = 1 / (1 + np.exp(0.1 * distance - 1))
            
#             # Angle modifier
#             angle_mod = np.sin(np.radians(angle))
            
#             # Situation modifier
#             situation_mods = {
#                 'penalty': 0.76,
#                 'set_piece': 0.08,
#                 'open_play': 0.12,
#                 'counter': 0.15
#             }
            
#             return base_prob * angle_mod * situation_mods.get(situation, 0.1)
        
#         xG = shots_data.apply(
#             lambda x: shot_probability(
#                 x['distance'],
#                 x['angle'],
#                 x['situation']
#             ),
#             axis=1
#         ).sum()
        
#         return xG

#     @staticmethod
#     def calculate_ppda(team_data: pd.DataFrame) -> float:
#         """
#         Calculate Passes Allowed Per Defensive Action (PPDA)
#         Lower PPDA = Higher pressing intensity
#         """
#         defensive_actions = (
#             team_data['tackles'].sum() +
#             team_data['interceptions'].sum() +
#             team_data['fouls'].sum()
#         )
        
#         opponent_passes = team_data['opponent_passes'].sum()
        
#         return opponent_passes / defensive_actions if defensive_actions > 0 else 0

#     @staticmethod
#     def calculate_field_tilt(possession_data: pd.DataFrame) -> float:
#         """
#         Calculate Field Tilt (percentage of possession in opposing third)
#         """
#         total_possession = possession_data['total_possession_time'].sum()
#         opp_third_time = possession_data['time_in_opp_third'].sum()
        
#         return (opp_third_time / total_possession * 100) if total_possession > 0 else 0

#     @staticmethod
#     def calculate_shot_quality(shots_data: pd.DataFrame) -> Dict[str, float]:
#         """
#         Analyze shot quality metrics
#         """
#         total_shots = len(shots_data)
#         if total_shots == 0:
#             return {'shot_quality': 0, 'shot_conversion': 0}
            
#         shots_on_target = len(shots_data[shots_data['on_target'] == True])
#         goals = len(shots_data[shots_data['result'] == 'goal'])
        
#         return {
#             'shot_quality': shots_on_target / total_shots,
#             'shot_conversion': goals / total_shots if total_shots > 0 else 0
#         }

# class EnhancedFootballPredictor(FootballPredictor):
#     def __init__(self, api_key: str):
#         super().__init__(api_key)
#         self.metrics_calculator = AdvancedMetricsCalculator()
        
#         # Additional prediction models
#         self.xg_model = RandomForestRegressor(n_estimators=200, random_state=42)
#         self.shots_model = RandomForestRegressor(n_estimators=200, random_state=42)
#         self.ppda_model = RandomForestRegressor(n_estimators=200, random_state=42)
#         self.field_tilt_model = RandomForestRegressor(n_estimators=200, random_state=42)
        
#     def prepare_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Enhanced feature preparation including advanced metrics"""
#         # Get basic features from parent class
#         features = super().prepare_advanced_features(data)
        
#         # Add advanced metrics
#         for idx, match in data.iterrows():
#             home_team = match['home_team']
#             away_team = match['away_team']
            
#             # Get recent matches for both teams
#             home_recent = data[
#                 (data['home_team'] == home_team) | 
#                 (data['away_team'] == home_team)
#             ].sort_values('date').tail(5)
            
#             away_recent = data[
#                 (data['home_team'] == away_team) | 
#                 (data['away_team'] == away_team)
#             ].sort_values('date').tail(5)
            
#             # Calculate advanced metrics
#             home_xG = self.metrics_calculator.calculate_xG(
#                 home_recent['shots_data'].iloc[-1]
#             )
#             away_xG = self.metrics_calculator.calculate_xG(
#                 away_recent['shots_data'].iloc[-1]
#             )
            
#             home_ppda = self.metrics_calculator.calculate_ppda(home_recent)
#             away_ppda = self.metrics_calculator.calculate_ppda(away_recent)
            
#             home_field_tilt = self.metrics_calculator.calculate_field_tilt(
#                 home_recent['possession_data'].iloc[-1]
#             )
#             away_field_tilt = self.metrics_calculator.calculate_field_tilt(
#                 away_recent['possession_data'].iloc[-1]
#             )
            
#             home_shot_metrics = self.metrics_calculator.calculate_shot_quality(
#                 home_recent['shots_data'].iloc[-1]
#             )
#             away_shot_metrics = self.metrics_calculator.calculate_shot_quality(
#                 away_recent['shots_data'].iloc[-1]
#             )
            
#             # Add new features
#             new_features = {
#                 'home_xG': home_xG,
#                 'away_xG': away_xG,
#                 'home_ppda': home_ppda,
#                 'away_ppda': away_ppda,
#                 'home_field_tilt': home_field_tilt,
#                 'away_field_tilt': away_field_tilt,
#                 'home_shot_quality': home_shot_metrics['shot_quality'],
#                 'away_shot_quality': away_shot_metrics['shot_quality'],
#                 'home_shot_conversion': home_shot_metrics['shot_conversion'],
#                 'away_shot_conversion': away_shot_metrics['shot_conversion']
#             }
            
#             # Update features DataFrame
#             features.loc[idx, new_features.keys()] = new_features.values()
            
#         return features

#     def predict_match(self, home_team: str, away_team: str, match_date: datetime) -> Dict:
#         """Enhanced match predictions including advanced metrics"""
#         # Get basic predictions
#         predictions = super().predict_match(home_team, away_team, match_date)
        
#         # Add advanced metric predictions
#         features = self.prepare_advanced_features(self.get_recent_matches(home_team, away_team))
#         match_features = features.iloc[-1].values.reshape(1, -1)
#         match_features_scaled = self.scaler.transform(match_features)
        
#         # Make additional predictions
#         advanced_predictions = {
#             'home_xG': self.xg_model.predict(match_features_scaled)[0],
#             'away_xG': self.xg_model.predict(match_features_scaled)[0],
#             'total_shots': self.shots_model.predict(match_features_scaled)[0],
#             'home_ppda': self.ppda_model.predict(match_features_scaled)[0],
#             'away_ppda': self.ppda_model.predict(match_features_scaled)[0],
#             'home_field_tilt': self.field_tilt_model.predict(match_features_scaled)[0],
#             'away_field_tilt': self.field_tilt_model.predict(match_features_scaled)[0]
#         }
        
#         predictions.update(advanced_predictions)
#         return predictions

#     def display_detailed_prediction(self, home_team: str, away_team: str, 
#                                   prediction: Dict, match_date: datetime) -> None:
#         """Enhanced prediction display including advanced metrics"""
#         # Display basic predictions
#         super().display_detailed_prediction(home_team, away_team, prediction, match_date)
        
#         # Display advanced metrics
#         advanced_metrics = [
#             ['Expected Goals (xG)', f"{home_team}: {prediction['home_xG']:.2f} | "
#                                   f"{away_team}: {prediction['away_xG']:.2f}"],
#             ['Total Shots', f"{prediction['total_shots']:.1f}"],
#             ['Pressing Intensity (PPDA)', f"{home_team}: {prediction['home_ppda']:.1f} | "
#                                         f"{away_team}: {prediction['away_ppda']:.1f}"],
#             ['Field Tilt %', f"{home_team}: {prediction['home_field_tilt']:.1f}% | "
#                             f"{away_team}: {prediction['away_field_tilt']:.1f}%"]
#         ]
        
#         print(Fore.CYAN + "\nAdvanced Metrics:" + Style.RESET_ALL)
#         print(tabulate(advanced_metrics, tablefmt='grid'))
        
#         # Display tactical insights
#         self._display_tactical_insights(prediction)

#     def _display_tactical_insights(self, prediction: Dict) -> None:
#         """Display insights based on advanced metrics"""
#         print(Fore.CYAN + "\nTactical Insights:" + Style.RESET_ALL)
        
#         # xG Analysis
#         xg_diff = prediction['home_xG'] - prediction['away_xG']
#         if abs(xg_diff) > 0.5:
#             dominant_team = "Home" if xg_diff > 0 else "Away"
#             print(f"📊 {dominant_team} team shows superior chance creation")
        
#         # Pressing Analysis
#         home_pressing = prediction['home_ppda'] < 10
#         away_pressing = prediction['away_ppda'] < 10
#         if home_pressing and away_pressing:
#             print("⚔️ High-intensity pressing game expected from both teams")
#         elif home_pressing:
#             print("⚔️ Home team likely to employ high pressing tactics")
#         elif away_pressing:
#             print("⚔️ Away team likely to employ high pressing tactics")
        
#         # Field Tilt Analysis
#         if abs(prediction['home_field_tilt'] - prediction['away_field_tilt']) > 10:
#             dominant_team = "Home" if prediction['home_field_tilt'] > prediction['away_field_tilt'] else "Away"
#             print(f"🎯 {dominant_team} team expected to dominate territorial possession")
        
#         # Shot Quality Analysis
#         if prediction['total_shots'] > 25:
#             print("🎯 High-volume shooting match expected")

# if __name__ == "__main__":
#     # Initialize predictor with API key
#     API_KEY = "aac7116ce0e240ae92e90fc35e4beb4b"
#     predictor = EnhancedFootballPredictor(API_KEY)
    
#     # Example prediction
#     home_team = "AC Milan W"
#     away_team = "Como W"
#     match_date = datetime.now()
    
#     # Get and display prediction
#     prediction = predictor.predict_match(home_team, away_team, match_date)
#     predictor.display_detailed_prediction(home_team, away_team, prediction, match_date)