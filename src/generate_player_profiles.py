import os
import pandas as pd

def get_latest_stats(df):
    """
    Calcula estadisticas avanzadas y se queda con el estado mas reciente de cada jugador.
    """
    # Limpieza extrema de columnas
    df.columns = [c.lower().strip().strip("'").strip('"') for c in df.columns]
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

    # Crear dataset vertical
    p1 = df[['tourney_date', 'winner_id', 'winner_name', 'winner_rank', 'winner_age', 'winner_ht', 'winner_hand', 'surface']].copy()
    p1.columns = ['date', 'id', 'name', 'rank', 'age', 'ht', 'hand', 'surface']
    p1['won'] = 1

    p2 = df[['tourney_date', 'loser_id', 'loser_name', 'loser_rank', 'loser_age', 'loser_ht', 'loser_hand', 'surface']].copy()
    p2.columns = ['date', 'id', 'name', 'rank', 'age', 'ht', 'hand', 'surface']
    p2['won'] = 0

    all_players = pd.concat([p1, p2]).sort_values(['date', 'id'])
    
    # FORMA: Ultimos 10 partidos
    all_players['form'] = all_players.groupby('id')['won'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    
    # EFECTIVIDAD POR SUPERFICIE (Hard, Clay, Grass)
    for s in ['Hard', 'Clay', 'Grass']:
        mask = all_players['surface'] == s
        all_players.loc[mask, f'eff_{s.lower()}'] = all_players[mask].groupby('id')['won'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        # Forward fill para que el jugador tenga su efectividad en esa superficie aunque el ultimo partido fuera en otra
        all_players[f'eff_{s.lower()}'] = all_players.groupby('id')[f'eff_{s.lower()}'].ffill().fillna(0.5)

    # Quedarse con el registro mas reciente por jugador
    latest = all_players.sort_values('date', ascending=False).drop_duplicates('name')
    
    return latest[['name', 'id', 'rank', 'age', 'ht', 'hand', 'form', 'eff_hard', 'eff_clay', 'eff_grass', 'date']]

def generate_profiles():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    atp_in = os.path.join(processed_dir, "atp_matches_combined.csv")
    wta_in = os.path.join(processed_dir, "wta_matches_combined.csv")
    
    profiles = []
    
    for file_path in [atp_in, wta_in]:
        if os.path.exists(file_path):
            print(f"Generando perfiles Pro desde {os.path.basename(file_path)}...")
            df = pd.read_csv(file_path)
            stats = get_latest_stats(df)
            profiles.append(stats)
            
    if not profiles:
        print("No se encontraron archivos de datos.")
        return
        
    final_profiles = pd.concat(profiles, ignore_index=True)
    # Resolver duplicados finales (jugadores que saltan de circuito o errores de ID)
    final_profiles = final_profiles.sort_values(by='date', ascending=False).drop_duplicates(subset=['name'])
    
    out_path = os.path.join(processed_dir, "player_profiles.csv")
    final_profiles.to_csv(out_path, index=False)
    print(f"Perfiles Pro de {len(final_profiles)} jugadores guardados con exito.")

if __name__ == "__main__":
    generate_profiles()

if __name__ == "__main__":
    generate_profiles()
    
