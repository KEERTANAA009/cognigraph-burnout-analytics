import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_complex_model():
    df = pd.read_csv('data/complex_student_data.csv')
    
    features = [
        'semantic_fatigue_score', 'help_seeking_drop_pct', 
        'forum_sentiment', 'lms_logins', 
        'assignment_delay_days', 'peer_contagion_risk'
    ]
    
    X = df[features]
    y = df['final_burnout_score']
    
    # We use a slightly deeper Random Forest for the complex data
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/cognigraph_model.pkl')
    print("CogniGraph model trained and saved.")

if __name__ == "__main__":
    train_complex_model()