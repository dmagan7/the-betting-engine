import os
import pandas as pd
import numpy as np
import re

def calculate_rolling_stats(df):
    """
    Calcula estadisticas rodantes (forma, wins por superficie) de manera eficiente.
    """
    # Limpieza extrema de columnas
    df.columns = [c.lower().strip().strip("'").strip('"') for c in df.columns]
    
    required = ['tourney_date', 'match_num', 'winner_id', 'loser_id', 'surface']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Faltan columnas criticas: {missing}")
        print(f"Columnas disponibles: {df.columns.tolist()}")
        return df

    print(f"Procesando {len(df)} partidos...")
    
    # Asegurar orden cronologico
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

    # Crear un DataFrame vertical de jugadores para calcular su historial
    p1 = df[['tourney_date', 'winner_id', 'winner_rank', 'surface', 'tourney_level', 'winner_hand', 'winner_ht']].copy()
    p1.columns = ['date', 'player_id', 'rank', 'surface', 'level', 'hand', 'ht']
    p1['won'] = 1

    p2 = df[['tourney_date', 'loser_id', 'loser_rank', 'surface', 'tourney_level', 'loser_hand', 'loser_ht']].copy()
    p2.columns = ['date', 'player_id', 'rank', 'surface', 'level', 'hand', 'ht']
    p2['won'] = 0

    players_hist = pd.concat([p1, p2]).sort_values(['date', 'player_id'])
    
    # 1. Forma (ultimos 10 partidos)
    players_hist['form'] = players_hist.groupby('player_id')['won'].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())
    
    # 2. Victoria en superficie especifica
    players_hist['surface_win_rate'] = players_hist.groupby(['player_id', 'surface'])['won'].transform(lambda x: x.shift().rolling(20, min_periods=1).mean())
    
    # 3. Fatiga (dias desde ultimo partido)
    players_hist['last_match_date'] = players_hist.groupby('player_id')['date'].shift()
    players_hist['days_since_last'] = (players_hist['date'] - players_hist['last_match_date']).dt.days.fillna(30)
    
    # Unir de vuelta al DF original
    # Para el ganador
    winner_stats = players_hist[players_hist['won'] == 1].drop_duplicates(['date', 'player_id'], keep='last')
    df = df.merge(winner_stats[['date', 'player_id', 'form', 'surface_win_rate', 'days_since_last']], 
                  left_on=['tourney_date', 'winner_id'], right_on=['date', 'player_id'], how='left', suffixes=('', '_w'))
    
    # Para el perdedor
    loser_stats = players_hist[players_hist['won'] == 0].drop_duplicates(['date', 'player_id'], keep='last')
    df = df.merge(loser_stats[['date', 'player_id', 'form', 'surface_win_rate', 'days_since_last']], 
                  left_on=['tourney_date', 'loser_id'], right_on=['date', 'player_id'], how='left', suffixes=('_w', '_l'))
    
    return df

def create_features(df):
    print("Extrayendo caracteristicas avanzadas...")
    
    # 1. Calcular estadisticas rodantes
    df = calculate_rolling_stats(df)
    
    np.random.seed(42)
    swap = np.random.rand(len(df)) > 0.5
    
    df_ml = pd.DataFrame()
    df_ml['tourney_date'] = df['tourney_date']
    df_ml['p1_won'] = (~swap).astype(int)
    
    # Caracteristicas de Jugador 1
    df_ml['p1_rank'] = np.where(swap, df['loser_rank'], df['winner_rank'])
    df_ml['p1_age'] = np.where(swap, df['loser_age'], df['winner_age'])
    df_ml['p1_ht'] = np.where(swap, df['loser_ht'], df['winner_ht'])
    df_ml['p1_hand'] = np.where(swap, df['loser_hand'], df['winner_hand'])
    df_ml['p1_hand'] = df_ml['p1_hand'].map({'R': 0, 'L': 1, 'U': 0}).fillna(0)
    
    df_ml['p1_form'] = np.where(swap, df['form_l'], df['form_w'])
    df_ml['p1_surface_eff'] = np.where(swap, df['surface_win_rate_l'], df['surface_win_rate_w'])
    df_ml['p1_fatigue'] = np.where(swap, df['days_since_last_l'], df['days_since_last_w'])

    # Caracteristicas de Jugador 2
    df_ml['p2_rank'] = np.where(swap, df['winner_rank'], df['loser_rank'])
    df_ml['p2_age'] = np.where(swap, df['winner_age'], df['loser_age'])
    df_ml['p2_ht'] = np.where(swap, df['winner_ht'], df['loser_ht'])
    df_ml['p2_hand'] = np.where(swap, df['winner_hand'], df['loser_hand'])
    df_ml['p2_hand'] = df_ml['p2_hand'].map({'R': 0, 'L': 1, 'U': 0}).fillna(0)
    
    df_ml['p2_form'] = np.where(swap, df['form_w'], df['form_l'])
    df_ml['p2_surface_eff'] = np.where(swap, df['surface_win_rate_w'], df['surface_win_rate_l'])
    df_ml['p2_fatigue'] = np.where(swap, df['days_since_last_w'], df['days_since_last_l'])

    # Diferenciales (Muy importantes para ML)
    df_ml['rank_diff'] = df_ml['p2_rank'] - df_ml['p1_rank']
    df_ml['age_diff'] = df_ml['p1_age'] - df_ml['p2_age']
    df_ml['ht_diff'] = df_ml['p1_ht'] - df_ml['p2_ht']
    df_ml['form_diff'] = df_ml['p1_form'] - df_ml['p2_form']
    df_ml['surface_diff'] = df_ml['p1_surface_eff'] - df_ml['p2_surface_eff']

    # Tournament Level
    df_ml['tourney_level'] = df['tourney_level'].map({'G': 3, 'M': 2, 'A': 1, 'C': 0}).fillna(0)
    df_ml['best_of'] = df['best_of']

    # --- NUEVOS TARGETS PARA MULTI-MERCADO ---
    # Usamos la logica de parse_score (ya integrada en el flujo o la anadimos de nuevo)
    # Nota: p1_won ya esta arriba. Necesitamos total_games y exact_score.
    # Como calculate_rolling_stats devuelve el DF original 'df' con columnas extra, 
    # podemos extraer los datos de 'df'.
    
    import re
    def get_market_targets(row):
        score_str = row['score']
        p1_won = row['p1_won_target'] # Necesitamos el target real relativo a P1
        best_of = row['best_of']
        
        if not isinstance(score_str, str) or any(x in score_str for x in ['RET', 'W/O', 'DEF', 'Default', 'Walkover']):
            return pd.Series([-1, -1, -1])
        clean_score = re.sub(r'\(\d+\)', '', score_str)
        w_sets, l_sets, w_games, l_games = 0, 0, 0, 0
        for s in clean_score.split():
            if '-' in s:
                parts = s.split('-')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    gw, gl = int(parts[0]), int(parts[1])
                    w_games += gw; l_games += gl
                    if gw > gl: w_sets += 1
                    elif gl > gw: l_sets += 1
        if (w_games + l_games) == 0: return pd.Series([-1, -1, -1])
        
        # p1_won_target es 1 si el p1 de la fila ML gano
        p1_sets = w_sets if p1_won == 1 else l_sets
        p2_sets = l_sets if p1_won == 1 else w_sets
        
        exact_score_class = -1
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
        return pd.Series([exact_score_class, w_games + l_games, w_sets + l_sets])

    # Marcamos p1_won en el DF original para facilitar el mapeo
    df['p1_won_target'] = (~swap).astype(int)
    df_ml[['exact_score', 'total_games', 'total_sets']] = df.apply(get_market_targets, axis=1)

    # Limpiamos anomalias
    df_ml = df_ml[df_ml['exact_score'] != -1]
    df_ml = df_ml.fillna(0)
    
    return df_ml

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    atp_in = os.path.join(processed_dir, "atp_matches_combined.csv")
    wta_in = os.path.join(processed_dir, "wta_matches_combined.csv")
    
    if os.path.exists(atp_in):
        print(f"DEBUG: Leyendo {atp_in}")
        atp = pd.read_csv(atp_in)
        print(f"DEBUG: Columnas: {atp.columns.tolist()}")
        atp_feat = create_features(atp)
        out = os.path.join(processed_dir, "atp_features.csv")
        atp_feat.to_csv(out, index=False)
        print(f"Features Pro ATP guardadas: {atp_feat.shape}")
        
    if os.path.exists(wta_in):
        wta = pd.read_csv(wta_in)
        wta_feat = create_features(wta)
        out = os.path.join(processed_dir, "wta_features.csv")
        wta_feat.to_csv(out, index=False)
        print(f"Features Pro WTA guardadas: {wta_feat.shape}")

if __name__ == "__main__":
    main()
