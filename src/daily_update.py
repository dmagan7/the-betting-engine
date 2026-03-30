import os
import subprocess
import sys

def run_command(command, cwd=None):
    print(f"-> Ejecutando: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error ({result.returncode}): {result.stderr}")
    else:
        print("OK.")
    return result.returncode == 0

def main():
    base_dir = r"C:\Users\PcVIP\.gemini\antigravity\scratch\tennis_prediction_ai"
    # Si estamos en Docker/Linux, la ruta base es el WORKDIR actual
    if not os.path.exists(base_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    atp_dir = os.path.join(base_dir, "data", "raw", "tennis_atp")
    wta_dir = os.path.join(base_dir, "data", "raw", "tennis_wta")
    
    # Asegurarnos de que las carpetas padre existan
    os.makedirs(os.path.dirname(atp_dir), exist_ok=True)
    
    print("=== RUTINA DE AUTOMATIZACIÓN DIARIA (MLOps) ===")
    print("1. Descargando / Actualizando resultados de GitHub...")
    
    # Si es la primera vez que se lanza en Docker, usar clone en lugar de pull
    if os.path.exists(atp_dir) and os.path.exists(os.path.join(atp_dir, ".git")):
        run_command(["git", "pull"], cwd=atp_dir)
    else:
        print("Primera instalación detectada. Clonando ATP desde cero...")
        run_command(["git", "clone", "https://github.com/JeffSackmann/tennis_atp.git", atp_dir])
        
    if os.path.exists(wta_dir) and os.path.exists(os.path.join(wta_dir, ".git")):
        run_command(["git", "pull"], cwd=wta_dir)
    else:
        print("Primera instalación detectada. Clonando WTA desde cero...")
        run_command(["git", "clone", "https://github.com/JeffSackmann/tennis_wta.git", wta_dir])
    
    python_exe = sys.executable
    
    print("\n2. Consolidando base de datos y limpiando features antiguas...")
    run_command([python_exe, "src/data_processing.py"], cwd=base_dir)
    
    print("\n3. Calculando métricas Multi-Mercado y Labels...")
    run_command([python_exe, "src/feature_engineering.py"], cwd=base_dir)
    
    print("\n4. Reentrenando Inteligencia Artificial (Gradient Boosting & Poisson)...")
    run_command([python_exe, "src/train_model.py"], cwd=base_dir)
    
    print("\n✅ Actualización diaria de la IA finalizada exitosamente.")

if __name__ == "__main__":
    main()
