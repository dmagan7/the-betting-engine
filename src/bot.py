import os
import joblib
import pandas as pd
import numpy as np
import datetime
import sys
import subprocess
import certifi
import requests
from scipy.stats import poisson
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
try:
    atp_score_model = joblib.load(os.path.join(MODELS_DIR, 'atp_exact_score_model.pkl'))
    atp_games_model = joblib.load(os.path.join(MODELS_DIR, 'atp_total_games_model.pkl'))
except:
    atp_score_model = None
    atp_games_model = None

def calculate_kelly(prob, odds, fraction=0.25, max_stake=5.0):
    if prob == 0 or odds <= 1: return 0.0
    q = 1 - prob
    b = odds - 1
    kelly_pct = ((b * prob) - q) / b
    if kelly_pct <= 0: return 0.0
    return min(kelly_pct * fraction * 100, max_stake)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👑 The Betting Engine Live. Usa /valuebets para escanear Bet365 en tiempo real.")

async def daily_update_job(context: ContextTypes.DEFAULT_TYPE):
    print("--------------------------------------------------")
    print("[CRON JOB] Iniciando rutina 24H de automatización MLOps...")
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), 'daily_update.py')
    subprocess.run([python_exe, script_path])
    global atp_score_model, atp_games_model
    try:
        atp_score_model = joblib.load(os.path.join(MODELS_DIR, 'atp_exact_score_model.pkl'))
        atp_games_model = joblib.load(os.path.join(MODELS_DIR, 'atp_total_games_model.pkl'))
        print("[CRON JOB] Modelos re-cargados en caliente.")
    except Exception as e:
        print(f"[CRON JOB] Error recargando modelos: {e}")
    print("--------------------------------------------------")

async def valuebets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 Conectando de verdad con API REST Bet365 para bajar partidos hoy...")
    ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "547f9d4f748137ff6adbf7fe48baa1a7")
    url = "https://api.the-odds-api.com/v4/sports/tennis_atp/odds/"
    params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h,totals", "bookmakers": "bet365"}
    
    try:
        resp = requests.get(url, params=params)
        matches = resp.json()
    except Exception as e:
        await update.message.reply_text(f"❌ Error conectando servidor API REST: {e}")
        return
        
    if not isinstance(matches, list) or len(matches) == 0:
        await update.message.reply_text("❌ No hay partidos en la base de datos oficial de The Odds API actualmente.")
        return

    responses = ["💰 *Escaneo de Cuotas Bet365 (Live HTTP)* 💰\n"]
    found_any = False
    
    # Evaluamos hasta 30 partidos paralelos
    for m in matches[:30]:
        if not m.get('bookmakers'): continue
        bm = m['bookmakers'][0] # Estamos seguros de que es bet365 por la API filter
        
        p1_name = m['home_team']
        p2_name = m['away_team']
        p1_odds, p2_odds, over_odds, over_line = 0, 0, 0, 21.5
        
        for market in bm['markets']:
            if market['key'] == 'h2h':
                for out in market['outcomes']:
                    if out['name'] == p1_name: p1_odds = out['price']
                    else: p2_odds = out['price']
            elif market['key'] == 'totals':
                for out in market['outcomes']:
                    if out['name'] == 'Over':
                        over_line = out.get('point', 21.5)
                        over_odds = out['price']
                        
        if p1_odds == 0 or p2_odds == 0: continue
        
        # Predictor IA en base a jugador basico/general
        df_pred = pd.DataFrame([{
            'surface_target': 0, 'p1_rank': 150, 'p1_age': 26, 'p1_ht': 185,
            'p2_rank': 150, 'p2_age': 26, 'p2_ht': 185
        }])
        
        score_probs = atp_score_model.predict_proba(df_pred)[0]
        prob_p1_win = score_probs[0] + score_probs[1]
        
        texto = f"\n🔹 *{p1_name} vs {p2_name}*\n"
        bets_found = False
        
        edge_p1 = (prob_p1_win * p1_odds) - 1
        if edge_p1 > 0.03:
            stake = calculate_kelly(prob_p1_win, p1_odds)
            texto += f"✅ Gana *{p1_name}* | Cuota: {p1_odds}\n"
            texto += f"   └ Edge: +{edge_p1*100:.1f}% | Stake Kelly: *{stake:.1f} U.*\n"
            bets_found = True
            
        prob_p2_win = score_probs[2] + score_probs[3]
        edge_p2 = (prob_p2_win * p2_odds) - 1
        if edge_p2 > 0.03:
            stake = calculate_kelly(prob_p2_win, p2_odds)
            texto += f"✅ Gana *{p2_name}* | Cuota: {p2_odds}\n"
            texto += f"   └ Edge: +{edge_p2*100:.1f}% | Stake Kelly: *{stake:.1f} U.*\n"
            bets_found = True
            
        if over_odds > 0:
            expected_games = atp_games_model.predict(df_pred)[0]
            prob_over = 1 - poisson.cdf(int(over_line), mu=expected_games)
            edge_over = (prob_over * over_odds) - 1
            if edge_over > 0.03:
                stake = calculate_kelly(prob_over, over_odds)
                texto += f"✅ Juegos > {over_line} | Cuota: {over_odds}\n"
                texto += f"   └ Edge: +{edge_over*100:.1f}% | Stake Kelly: *{stake:.1f} U.*\n"
                bets_found = True
                
        if bets_found:
            responses.append(texto)
            found_any = True
            
    if not found_any:
        responses.append("\n❌ Analizados 30 partidos de la API, pero ninguno ofrece Cuotas con EV+ Positivo frente a nuestra IA.")
        
    msg = "\n".join(responses)
    for i in range(0, len(msg), 4000):
        await update.message.reply_text(msg[i:i+4000], parse_mode='Markdown')

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token or token.startswith("tu_token"):
        print("ERROR: Abre el archivo .env e inserta tu Token de BotFather.")
        return
        
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("valuebets", valuebets))

    t = datetime.time(hour=4, minute=0, tzinfo=datetime.timezone.utc)
    application.job_queue.run_daily(daily_update_job, t)

    print("Motor API Bot Iniciado. En Espera...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
