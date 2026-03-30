import os
import pandas as pd
import numpy as np
import re

def parse_score(row):
    score_str = row['score']
    p1_won = row['p1_won']
    best_of = row['best_of']
    
    if not isinstance(score_str, str) or any(x in score_str for x in ['RET', 'W/O', 'DEF', 'Default', 'Walkover']):
        return pd.Series([-1, -1, -1])
        
    # Limpiar tiebreaks, ej: "7-6(5)" -> "7-6"
    clean_score = re.sub(r'\(\d+\)', '', score_str)
    
    w_sets, l_sets, w_games, l_games = 0, 0, 0, 0
    for s in clean_score.split():
        if '-' in s:
            parts = s.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                gw, gl = int(parts[0]), int(parts[1])
                w_games += gw
                l_games += gl
                if gw > gl: w_sets += 1
                elif gl > gw: l_sets += 1
                
    total_games = w_games + l_games
    total_sets = w_sets + l_sets
    
    # Ignorar partidos sin completar
    if total_games == 0 or (w_sets == 0 and l_sets == 0):
        return pd.Series([-1, -1, -1])
        
    p1_sets = w_sets if p1_won == 1 else l_sets
    p2_sets = l_sets if p1_won == 1 else w_sets
    
    exact_score_class = -1
    # Clases multi-mercado:
    # Bo3: 0: 2-0, 1: 2-1, 2: 0-2, 3: 1-2
    # Bo5: 4: 3-0, 5: 3-1, 6: 3-2, 7: 0-3, 8: 1-3, 9: 2-3
    if best_of == 3:
        if p1_sets == 2 and p2_sets == 0: exact_score_class = 0
        elif p1_sets == 2 and p2_sets == 1: exact_score_class = 1
        elif p1_sets == 0 and p2_sets == 2: exact_score_class = 2
        elif p1_sets == 1 and p2_sets == 2: exact_score_class = 3
    elif best_of == 5:
        if p1_sets == 3 and p2_sets == 0: exact_score_class = 4
        elif p1_sets == 3 and p2_sets == 1: exact_score_class = 5
        elif p1_sets == 3 and p2_sets == 2: exact_score_class = 6
        elif p1_sets == 0 and p2_sets == 3: exact_score_class = 7
        elif p1_sets == 1 and p2_sets == 3: exact_score_class = 8
        elif p1_sets == 2 and p2_sets == 3: exact_score_class = 9

    return pd.Series([exact_score_class, total_games, total_sets])

def create_features(df):
    print("Creando características Multi-Mercado para Bet365...")
    np.random.seed(42)
    swap = np.random.rand(len(df)) > 0.5
    
    df_ml = pd.DataFrame()
    df_ml['tourney_date'] = df['tourney_date']
    df_ml['best_of'] = df['best_of']
    df_ml['score'] = df['score']
    df_ml['surface_target'] = df['surface'].map({'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}).fillna(-1)
    
    # Target Binario de Winner
    df_ml['p1_won'] = (~swap).astype(int)
    
    df_ml['p1_rank'] = np.where(swap, df['loser_rank'], df['winner_rank'])
    df_ml['p1_age'] = np.where(swap, df['loser_age'], df['winner_age'])
    df_ml['p1_ht'] = np.where(swap, df['loser_ht'], df['winner_ht'])
    
    df_ml['p2_rank'] = np.where(swap, df['winner_rank'], df['loser_rank'])
    df_ml['p2_age'] = np.where(swap, df['winner_age'], df['loser_age'])
    df_ml['p2_ht'] = np.where(swap, df['winner_ht'], df['loser_ht'])
    
    # Extraer variables multi-mercado desde el String de Scores
    print("Parseando sets y juegos de decenas de miles de partidos históricos...")
    df_ml[['exact_score', 'total_games', 'total_sets']] = df_ml.apply(parse_score, axis=1)
    
    # Limpiamos anomalías (Scoreboards no resueltos o Walkovers)
    df_ml = df_ml[df_ml['exact_score'] != -1]
    
    # Rellenamos nulos numéricos (Features de jugadores)
    df_ml = df_ml.fillna(-999)
    df_ml = df_ml.drop(columns=['score'])
    
    return df_ml

def main():
    base_dir = r"C:\Users\PcVIP\.gemini\antigravity\scratch\tennis_prediction_ai"
    processed_dir = os.path.join(base_dir, "data", "processed")
    atp_in = os.path.join(processed_dir, "atp_matches_combined.csv")
    wta_in = os.path.join(processed_dir, "wta_matches_combined.csv")
    
    if os.path.exists(atp_in):
        atp = pd.read_csv(atp_in)
        atp_feat = create_features(atp)
        out = os.path.join(processed_dir, "atp_features.csv")
        atp_feat.to_csv(out, index=False)
        print(f"Features Multi-Mercado ATP guardadas: {atp_feat.shape}")
        
    if os.path.exists(wta_in):
        wta = pd.read_csv(wta_in)
        wta_feat = create_features(wta)
        out = os.path.join(processed_dir, "wta_features.csv")
        wta_feat.to_csv(out, index=False)
        print(f"Features Multi-Mercado WTA guardadas: {wta_feat.shape}")

if __name__ == "__main__":
    main()
