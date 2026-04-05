import os
import sys
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# 1. Configuración de logging INMEDIATA
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# 2. Servidor de Salud ULTRA-RÁPIDO para Azure (Antes que pandas/numpy)
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"OK - The Betting Engine is running.")
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args): return

def run_health_server():
    # Azure a veces usa WEBSITES_PORT en lugar de PORT
    port_str = os.environ.get("WEBSITES_PORT") or os.environ.get("PORT") or "80"
    try:
        port = int(port_str)
        server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
        logger.info(f"LOG: SALUD : Servidor HTTP de control activo en puerto {port}")
        
        def heartbeat():
            while True:
                logger.info("LOG: HEARTBEAT : El bot y el servidor de salud siguen vivos.")
                threading.Event().wait(120) # Reducimos frecuencia para no saturar logs
                
        threading.Thread(target=heartbeat, daemon=True).start()
        server.serve_forever()
    except Exception as e:
        logger.error(f"LOG: ERROR : Fallo crítico en Servidor de Salud: {e}")

# LANZAR SALUD YA MISMO (Hilo no-bloqueante)
logger.info("LOG: STARTUP : Iniciando health check server...")
threading.Thread(target=run_health_server, daemon=True).start()

logger.info("LOG: STARTUP : Cargando librerías pesadas (pandas, sklearn, etc)...")

import joblib
import pandas as pd
import numpy as np
import datetime
import subprocess
import certifi
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
from scipy.stats import poisson

load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
logger.info("LOG: SSL_CERT_FILE configurado")

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
try:
    if os.path.exists(os.path.join(MODELS_DIR, 'atp_exact_score_model.pkl')):
        atp_score_model = joblib.load(os.path.join(MODELS_DIR, 'atp_exact_score_model.pkl'))
        atp_games_model = joblib.load(os.path.join(MODELS_DIR, 'atp_total_games_model.pkl'))
        logger.info("Modelos cargados exitosamente.")
    else:
        logger.warning("No se encontraron archivos de modelos .pkl. Se requiere entrenamiento.")
        atp_score_model = None
        atp_games_model = None
except Exception as e:
    logger.error(f"Error cargando modelos: {e}")
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
    # 1. Obtener los deportes activos
    sports_url = "https://api.the-odds-api.com/v4/sports"
    try:
        logger.info("Obteniendo deportes de The Odds API...")
        sports_resp = requests.get(sports_url, params={"apiKey": ODDS_API_KEY}, timeout=10)
        sports_resp.raise_for_status()
        sports_data = sports_resp.json()
        
        tennis_sports = [s['key'] for s in sports_data if 'tennis' in s['key'].lower()]
        logger.info(f"Deportes de tenis encontrados: {len(tennis_sports)}")
    except Exception as e:
        logger.error(f"Error conectando a The Odds API (Sports): {e}")
        await update.message.reply_text(f"❌ Error conectando servidor API REST para obtener deportes: {e}")
        return

    if not tennis_sports:
        await update.message.reply_text("❌ No hay torneos de tenis activos en The Odds API actualmente.")
        return

    matches = []
    # 2. Descargar cuotas para cada torneo de tenis activo
    for sport_key in tennis_sports:
        odds_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h,totals", "bookmakers": "bet365"}
        try:
            resp = requests.get(odds_url, params=params, timeout=10)
            resp.raise_for_status()
            sport_matches = resp.json()
            if isinstance(sport_matches, list):
                matches.extend(sport_matches)
        except Exception as e:
            logger.warning(f"Error descargando cuotas para {sport_key}: {e}")
            continue

    if not matches:
        await update.message.reply_text("❌ No hay partidos en la base de datos oficial de The Odds API actualmente.")
        return

    # Verificación de modelos antes de procesar
    if atp_score_model is None or atp_games_model is None:
        logger.error("Los modelos de IA no están cargados. Abortando scan.")
        await update.message.reply_text("⚠️ ERROR: Los modelos de IA no se han cargado correctamente. El sistema necesita un reentrenamiento mediante /daily_update (manual o automático).")
        return

    responses = ["💰 *Escaneo de Cuotas Bet365 (Live HTTP)* 💰\n"]
    found_any = False
    logger.info(f"Procesando {min(len(matches), 30)} partidos...")
    
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
    try:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if token:
            token = token.strip('"').strip("'").strip()
        
        if not token or token.startswith("tu_token"):
            logger.error("LOG: FATAL : TELEGRAM_BOT_TOKEN no configurado en Azure App Settings.")
            logger.error("LOG: FATAL : El contenedor se cerrará ahora para evitar bucles de error.")
            return
        
        logger.info(f"LOG: STARTUP : Token validado (hash: {hash(token)})")
            
        application = Application.builder().token(token).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("valuebets", valuebets))

        t = datetime.time(hour=4, minute=0, tzinfo=datetime.timezone.utc)
        application.job_queue.run_daily(daily_update_job, t)

        logger.info("LOG: Motor API Bot Iniciado. run_polling...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"FATAL: El bot ha crasheado en main(): {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
