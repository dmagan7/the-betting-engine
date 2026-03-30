import os
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import joblib

def process_circuit(df, name, models_dir):
    print(f"\n--- Entrenando Modelos de Apuestas Multi-Mercado para {name} ---")
    df = df.sort_values(by='tourney_date')
    
    # Para apuestas precisas estandarizadas, modelaremos partidos al Mejor de 3 (Best of 3)
    df_bo3 = df[df['best_of'] == 3].copy()
    
    features = ['surface_target', 'p1_rank', 'p1_age', 'p1_ht', 'p2_rank', 'p2_age', 'p2_ht']
    X = df_bo3[features]
    
    # Target 1: Resultados Exactos / Sets. (0: 2-0, 1: 2-1, 2: 0-2, 3: 1-2)
    y_score = df_bo3['exact_score']
    # Target 2: Total de Juegos
    y_games = df_bo3['total_games']
    
    # Train test split cronológico
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    
    yscore_train = y_score.iloc[:split_idx]
    ygames_train = y_games.iloc[:split_idx]
    
    print("Entrenando Clasificador de Resultados Exactos Automático...")
    model_score = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, random_state=42, max_iter=150)
    model_score.fit(X_train, yscore_train)
    score_path = os.path.join(models_dir, f"{name.lower()}_exact_score_model.pkl")
    joblib.dump(model_score, score_path)
    
    print("Entrenando Regresor de Poisson para Total Juegos (Over/Under)...")
    # Utilizamos pérdida de Poisson ya que el total de juegos es una variable de "conteo"
    model_games = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.05, max_depth=6, random_state=42, max_iter=150)
    model_games.fit(X_train, ygames_train)
    games_path = os.path.join(models_dir, f"{name.lower()}_total_games_model.pkl")
    joblib.dump(model_games, games_path)
    
    print(f"-> Modelos de {name} guardados con éxito.")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    atp_in = os.path.join(processed_dir, "atp_features.csv")
    wta_in = os.path.join(processed_dir, "wta_features.csv")
    
    if os.path.exists(atp_in):
        process_circuit(pd.read_csv(atp_in), "ATP", models_dir)
        
    if os.path.exists(wta_in):
        process_circuit(pd.read_csv(wta_in), "WTA", models_dir)

if __name__ == "__main__":
    main()
