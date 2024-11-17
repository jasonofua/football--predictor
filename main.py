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
        
        # Additional prediction models
        self.xg_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.shots_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.ppda_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.field_tilt_model = RandomForestRegressor(n_estimators=200, random_state=42)
        
    def prepare_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature preparation including advanced metrics"""
        # Get basic features from parent class
        features = super().prepare_advanced_features(data)
        
        # Add advanced metrics
        for idx, match in data.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Get recent matches for both teams
            home_recent = data[
                (data['home_team'] == home_team) | 
                (data['away_team'] == home_team)
            ].sort_values('date').tail(5)
            
            away_recent = data[
                (data['home_team'] == away_team) | 
                (data['away_team'] == away_team)
            ].sort_values('date').tail(5)
            
            # Calculate advanced metrics
            home_xG = self.metrics_calculator.calculate_xG(
                home_recent['shots_data'].iloc[-1]
            )
            away_xG = self.metrics_calculator.calculate_xG(
                away_recent['shots_data'].iloc[-1]
            )
            
            home_ppda = self.metrics_calculator.calculate_ppda(home_recent)
            away_ppda = self.metrics_calculator.calculate_ppda(away_recent)
            
            home_field_tilt = self.metrics_calculator.calculate_field_tilt(
                home_recent['possession_data'].iloc[-1]
            )
            away_field_tilt = self.metrics_calculator.calculate_field_tilt(
                away_recent['possession_data'].iloc[-1]
            )
            
            home_shot_metrics = self.metrics_calculator.calculate_shot_quality(
                home_recent['shots_data'].iloc[-1]
            )
            away_shot_metrics = self.metrics_calculator.calculate_shot_quality(
                away_recent['shots_data'].iloc[-1]
            )
            
            # Add new features
            new_features = {
                'home_xG': home_xG,
                'away_xG': away_xG,
                'home_ppda': home_ppda,
                'away_ppda': away_ppda,
                'home_field_tilt': home_field_tilt,
                'away_field_tilt': away_field_tilt,
                'home_shot_quality': home_shot_metrics['shot_quality'],
                'away_shot_quality': away_shot_metrics['shot_quality'],
                'home_shot_conversion': home_shot_metrics['shot_conversion'],
                'away_shot_conversion': away_shot_metrics['shot_conversion']
            }
            
            # Update features DataFrame
            features.loc[idx, new_features.keys()] = new_features.values()
            
        return features

    def predict_match(self, home_team: str, away_team: str, match_date: datetime) -> Dict:
        """Enhanced match predictions including advanced metrics"""
        # Get basic predictions
        predictions = super().predict_match(home_team, away_team, match_date)
        
        # Add advanced metric predictions
        features = self.prepare_advanced_features(self.get_recent_matches(home_team, away_team))
        match_features = features.iloc[-1].values.reshape(1, -1)
        match_features_scaled = self.scaler.transform(match_features)
        
        # Make additional predictions
        advanced_predictions = {
            'home_xG': self.xg_model.predict(match_features_scaled)[0],
            'away_xG': self.xg_model.predict(match_features_scaled)[0],
            'total_shots': self.shots_model.predict(match_features_scaled)[0],
            'home_ppda': self.ppda_model.predict(match_features_scaled)[0],
            'away_ppda': self.ppda_model.predict(match_features_scaled)[0],
            'home_field_tilt': self.field_tilt_model.predict(match_features_scaled)[0],
            'away_field_tilt': self.field_tilt_model.predict(match_features_scaled)[0]
        }
        
        predictions.update(advanced_predictions)
        return predictions

    def display_detailed_prediction(self, home_team: str, away_team: str, 
                                  prediction: Dict, match_date: datetime) -> None:
        """Enhanced prediction display including advanced metrics"""
        # Display basic predictions
        super().display_detailed_prediction(home_team, away_team, prediction, match_date)
        
        # Display advanced metrics
        advanced_metrics = [
            ['Expected Goals (xG)', f"{home_team}: {prediction['home_xG']:.2f} | "
                                  f"{away_team}: {prediction['away_xG']:.2f}"],
            ['Total Shots', f"{prediction['total_shots']:.1f}"],
            ['Pressing Intensity (PPDA)', f"{home_team}: {prediction['home_ppda']:.1f} | "
                                        f"{away_team}: {prediction['away_ppda']:.1f}"],
            ['Field Tilt %', f"{home_team}: {prediction['home_field_tilt']:.1f}% | "
                            f"{away_team}: {prediction['away_field_tilt']:.1f}%"]
        ]
        
        print(Fore.CYAN + "\nAdvanced Metrics:" + Style.RESET_ALL)
        print(tabulate(advanced_metrics, tablefmt='grid'))
        
        # Display tactical insights
        self._display_tactical_insights(prediction)

    def _display_tactical_insights(self, prediction: Dict) -> None:
        """Display insights based on advanced metrics"""
        print(Fore.CYAN + "\nTactical Insights:" + Style.RESET_ALL)
        
        # xG Analysis
        xg_diff = prediction['home_xG'] - prediction['away_xG']
        if abs(xg_diff) > 0.5:
            dominant_team = "Home" if xg_diff > 0 else "Away"
            print(f"📊 {dominant_team} team shows superior chance creation")
        
        # Pressing Analysis
        home_pressing = prediction['home_ppda'] < 10
        away_pressing = prediction['away_ppda'] < 10
        if home_pressing and away_pressing:
            print("⚔️ High-intensity pressing game expected from both teams")
        elif home_pressing:
            print("⚔️ Home team likely to employ high pressing tactics")
        elif away_pressing:
            print("⚔️ Away team likely to employ high pressing tactics")
        
        # Field Tilt Analysis
        if abs(prediction['home_field_tilt'] - prediction['away_field_tilt']) > 10:
            dominant_team = "Home" if prediction['home_field_tilt'] > prediction['away_field_tilt'] else "Away"
            print(f"🎯 {dominant_team} team expected to dominate territorial possession")
        
        # Shot Quality Analysis
        if prediction['total_shots'] > 25:
            print("🎯 High-volume shooting match expected")

if __name__ == "__main__":
    # Initialize predictor with API key
    API_KEY = "your_api_key_here"
    predictor = EnhancedFootballPredictor(API_KEY)
    
    # Example prediction
    home_team = "Manchester City"
    away_team = "Liverpool"
    match_date = datetime.now()
    
    # Get and display prediction
    prediction = predictor.predict_match(home_team, away_team, match_date)
    predictor.display_detailed_prediction(home_team, away_team, prediction, match_date)