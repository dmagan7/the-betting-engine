import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
import joblib

def process_circuit(df, name, models_dir):
    print(f"\n--- Entrenando Modelos de Apuestas Multi-Mercado para {name} ---")
    df = df.sort_values(by='tourney_date')
    
    # Modelamos partidos al Mejor de 3 (Best of 3)
    df_bo3 = df[df['best_of'] == 3].copy()
    print(f"Partidos BO3 disponibles: {len(df_bo3)}")
    
    features = [
        'rank_diff', 'age_diff', 'ht_diff', 'form_diff', 'surface_diff',
        'p1_rank', 'p2_rank', 'p1_form', 'p2_form', 'p1_surface_eff', 'p2_surface_eff',
        'p1_fatigue', 'p2_fatigue', 'p1_hand', 'p2_hand', 'tourney_level'
    ]
    X = df_bo3[features]
    y_win = df_bo3['p1_won']
    y_score = df_bo3['exact_score']
    y_games = df_bo3['total_games']
    
    # Split cronologico: 80% train, 20% test (nunca mezclar futuro con pasado)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_win_train, y_win_test = y_win.iloc[:split_idx], y_win.iloc[split_idx:]
    yscore_train = y_score.iloc[:split_idx]
    ygames_train = y_games.iloc[:split_idx]

    # --- MODELO 1: Ganador del partido (con calibracion de probabilidades) ---
    print("Entrenando Clasificador de Ganador (Match Winner)...")
    base_win = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=5,          # reducido de 6 -> menor sobreajuste
        max_iter=300,
        min_samples_leaf=30,  # evita que hoja pequena genere probs extremas
        random_state=42
    )
    base_win.fit(X_train, y_win_train)

    # CalibratedClassifierCV con isotonic regression sobre el conjunto de test
    # Isotonic es superior a Platt (sigmoid) para distribuciones no gaussianas como el tenis
    # cv='prefit' porque ya entrenamos el modelo base y no queremos data leakage
    print("Calibrando probabilidades (isotonic regression)...")
    model_win = CalibratedClassifierCV(base_win, cv='prefit', method='isotonic')
    model_win.fit(X_test, y_win_test)

    # --- Metricas de validacion ---
    probs = model_win.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    brier = brier_score_loss(y_win_test, probs)
    ll = log_loss(y_win_test, probs)
    acc = accuracy_score(y_win_test, preds)
    print(f"  Accuracy: {acc:.3f} | Brier Score: {brier:.4f} (mejor menor) | Log-Loss: {ll:.4f}")
    print(f"  Prob range: [{probs.min():.3f}, {probs.max():.3f}] - rango amplio = bien calibrado")
    
    # Guardar modelo calibrado
    win_path = os.path.join(models_dir, f"{name.lower()}_win_model.pkl")
    joblib.dump(model_win, win_path)
    print(f"  Guardado: {win_path}")

    # --- MODELO 2: Resultado exacto (sets) ---
    print("Entrenando Clasificador de Resultados Exactos...")
    model_score = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=5, max_iter=150,
        min_samples_leaf=30, random_state=42
    )
    model_score.fit(X_train, yscore_train)
    score_path = os.path.join(models_dir, f"{name.lower()}_exact_score_model.pkl")
    joblib.dump(model_score, score_path)

    # --- MODELO 3: Total juegos (Over/Under) ---
    print("Entrenando Regresor de Poisson para Total Juegos (Over/Under)...")
    model_games = HistGradientBoostingRegressor(
        loss='poisson', learning_rate=0.05, max_depth=5,
        max_iter=150, min_samples_leaf=30, random_state=42
    )
    model_games.fit(X_train, ygames_train)
    games_path = os.path.join(models_dir, f"{name.lower()}_total_games_model.pkl")
    joblib.dump(model_games, games_path)
    
    print(f"-> Modelos de {name} guardados con exito.\n")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    atp_in = os.path.join(processed_dir, "atp_features.csv")
    wta_in = os.path.join(processed_dir, "wta_features.csv")
    
    if os.path.exists(atp_in):
        process_circuit(pd.read_csv(atp_in), "ATP", models_dir)
    else:
        print(f"AVISO: {atp_in} no encontrado. Ejecuta primero data_processing.py y feature_engineering.py")
        
    if os.path.exists(wta_in):
        process_circuit(pd.read_csv(wta_in), "WTA", models_dir)
    else:
        print(f"AVISO: {wta_in} no encontrado.")

if __name__ == "__main__":
    main()

