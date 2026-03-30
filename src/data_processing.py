import os
import pandas as pd

def load_and_combine(directory, prefix, start_year=2010, end_year=2024):
    df_list = []
    for year in range(start_year, end_year + 1):
        filename = f"{prefix}_{year}.csv"
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df_list.append(df)
            except Exception as e:
                print(f"Error leyendo {filepath}: {e}")
                
    if not df_list:
        return pd.DataFrame()
        
    combined = pd.concat(df_list, ignore_index=True)
    return combined

def clean_data(df):
    if df.empty:
        return df
    
    # Convertir fechas de formato YYYYMMDD a datetime
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    
    # Eliminar filas donde no haya IDs válidos para los jugadores
    df = df.dropna(subset=['winner_id', 'loser_id'])
    
    # Ordenar por fecha cronológicamente para evitar data leakage luego en ML
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    
    return df

def main():
    base_dir = r"C:\Users\PcVIP\.gemini\antigravity\scratch\tennis_prediction_ai"
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    atp_dir = os.path.join(raw_dir, "tennis_atp")
    wta_dir = os.path.join(raw_dir, "tennis_wta")
    
    print("Mapeando y procesando datos historicos ATP...")
    atp_df = load_and_combine(atp_dir, "atp_matches", 2010, 2024)
    atp_df = clean_data(atp_df)
    atp_out = os.path.join(processed_dir, "atp_matches_combined.csv")
    atp_df.to_csv(atp_out, index=False)
    print(f"-> Datos ATP consolidados en {atp_out}. Filas/Columnas: {atp_df.shape}")
    
    print("\nMapeando y procesando datos historicos WTA...")
    wta_df = load_and_combine(wta_dir, "wta_matches", 2010, 2024)
    wta_df = clean_data(wta_df)
    wta_out = os.path.join(processed_dir, "wta_matches_combined.csv")
    wta_df.to_csv(wta_out, index=False)
    print(f"-> Datos WTA consolidados en {wta_out}. Filas/Columnas: {wta_df.shape}")
    
    print("\nLimpieza y consolidacion completada exitosamente.")

if __name__ == "__main__":
    main()
