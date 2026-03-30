import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import poisson
import certifi
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Variables desde `.env` y certificados para peticiones (como Windows MSYS requiere a veces)
load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
try:
    atp_score_model = joblib.load(os.path.join(MODELS_DIR, 'atp_exact_score_model.pkl'))
    atp_games_model = joblib.load(os.path.join(MODELS_DIR, 'atp_total_games_model.pkl'))
except Exception as e:
    print(f"Error cargando los nuevos modelos multi-mercado: {e}")
    atp_score_model = None
    atp_games_model = None

def calculate_kelly(prob, odds, fraction=0.25, max_stake=5.0):
    """
    Calcula el Stake recomendado (en %) usando el Criterio de Kelly fraccional para proteger banca.
    """
    if prob == 0 or odds <= 1: return 0.0
    q = 1 - prob
    # Kelly Formula = (bp - q) / b, where b = odds - 1
    b = odds - 1
    kelly_pct = ((b * prob) - q) / b
    
    if kelly_pct <= 0: return 0.0 # No hay ventaja (EV negativo), no apostamos.
    
    stake = kelly_pct * fraction * 100
    return min(stake, max_stake)

async def daily_update_job(context: ContextTypes.DEFAULT_TYPE):
    print("--------------------------------------------------")
    print("[CRON JOB] Iniciando rutina 24H de MLOps...")
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), 'daily_update.py')
    # Ejecutamos el reentrenamiento
    subprocess.run([python_exe, script_path])
    
    # Recargar los modelos calientes en memoria para hoy
    global atp_score_model, atp_games_model
    try:
        atp_score_model = joblib.load(os.path.join(MODELS_DIR, 'atp_exact_score_model.pkl'))
        atp_games_model = joblib.load(os.path.join(MODELS_DIR, 'atp_total_games_model.pkl'))
        print("[CRON JOB] Modelos re-cargados con éxito para las apuestas frescas del día.")
    except Exception as e:
        print(f"[CRON JOB] Error recargando modelos nuevos: {e}")
    print("--------------------------------------------------")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👑 ¡Bienvenido a The Betting Engine (Edition: Bet365)!\n\n"
        "La Inteligencia Artificial acaba de ser configurada para Multi-Mercado.\n"
        "Usa el comando /valuebets para escanear las cuotas actuales contra mis pronósticos matemáticos."
    )
    await update.message.reply_text(welcome_text)

async def valuebets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 Analizando cuotas de Bet365 y cruzándolas con Algoritmos de IA...")
    
    if not atp_score_model or not atp_games_model:
        await update.message.reply_text("⚠ Modelos de IA no cargados. Revisa los logs de arranque.")
        return
        
    # MOCK API RESPONSE - Simulando obtener cuotas REST JSON de The Odds API para Bet365
    mock_matches = [
        {"p1": "Carlos Alcaraz", "p1_rank": 3, "p1_age": 21, "p1_ht": 183, 
         "p2": "Jannik Sinner", "p2_rank": 1, "p2_age": 22, "p2_ht": 188, "surface": 0,
         "odds": {"p1_win": 2.10, "p2_win": 1.72, "over_22_5": 1.95, "under_22_5": 1.83}}, 
         
        {"p1": "Novak Djokovic", "p1_rank": 2, "p1_age": 37, "p1_ht": 188, 
         "p2": "Daniil Medvedev", "p2_rank": 4, "p2_age": 28, "p2_ht": 198, "surface": 1,
         "odds": {"p1_win": 1.40, "p2_win": 3.00, "over_21_5": 1.90, "under_21_5": 1.90}}
    ]
    
    responses = ["💰 *Oportunidades de Inversión Identificadas* 💰\n"]
    
    for match in mock_matches:
        # Array Exacto de Feature Engineering V2
        df_pred = pd.DataFrame([{
            'surface_target': match['surface'],
            'p1_rank': match['p1_rank'],
            'p1_age': match['p1_age'],
            'p1_ht': match['p1_ht'],
            'p2_rank': match['p2_rank'],
            'p2_age': match['p2_age'],
            'p2_ht': match['p2_ht']
        }])
        
        # 1. MERCADOS DE RESULTADO EXACTO Y GANADOR
        score_probs = atp_score_model.predict_proba(df_pred)[0]
        # P(P1 Gana) = P(2-0) + P(2-1)
        prob_p1_win = score_probs[0] + score_probs[1]
        
        # 2. MERCADOS OVER/UNDER (Poisson Regressor)
        expected_games = atp_games_model.predict(df_pred)[0]
        # Usamos SciPy CDF para saber la probabilidad probabilística de que no se superen X juegos.
        prob_over_22_5 = 1 - poisson.cdf(22, mu=expected_games)
        prob_over_21_5 = 1 - poisson.cdf(21, mu=expected_games)
        
        texto = f"\n🔹 *{match['p1']} vs {match['p2']}*\n"
        bets_found = False
        
        # A) Analizamos Cuotas para Ganador del Partido
        edge_p1 = (prob_p1_win * match['odds']['p1_win']) - 1
        if edge_p1 > 0.03: # Existe ventaja Matemática contra Bet365 > 3%
            stake = calculate_kelly(prob_p1_win, match['odds']['p1_win'])
            texto += f"✅ *Victoria {match['p1']}*\n"
            texto += f"   └ Bet365: {match['odds']['p1_win']} | IA Prob Real: {prob_p1_win*100:.1f}%\n"
            texto += f"   └ Edge: +{edge_p1*100:.1f}% | 🎯 Stake: *{stake:.1f} U.*\n"
            bets_found = True
            
        # B) Analizamos Cuotas para Total Juegos
        linea = 22.5 if "Sinner" in match['p2'] else 21.5
        prob_over = prob_over_22_5 if linea == 22.5 else prob_over_21_5
        cuota_over = match['odds'][f"over_{str(linea).replace('.','_')}"]
        
        edge_over = (prob_over * cuota_over) - 1
        if edge_over > 0.03:
            stake = calculate_kelly(prob_over, cuota_over)
            texto += f"✅ *O/U {linea} Juegos (OVER)*\n"
            texto += f"   └ Bet365: {cuota_over} | IA Prob Real: {prob_over*100:.1f}%\n"
            texto += f"   └ Edge: +{edge_over*100:.1f}% | 🎯 Stake: *{stake:.1f} U.*\n"
            bets_found = True
            
        if not bets_found:
            texto += "❌ Sin Value Bets localizadas en Bet365 a cuota actual.\n"
            
        responses.append(texto)
        
    await update.message.reply_text("\n".join(responses), parse_mode='Markdown')

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token or token.startswith("tu_token"):
        print("ERROR: Abre el archivo .env e inserta tu Token de BotFather.")
        return
        
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("valuebets", valuebets))

    print("Motor de Value Bets Iniciado. Escaneando en Telegram...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
