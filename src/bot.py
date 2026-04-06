import sys, os
print("LOG: BOOT : Bot script started at the very first line.", flush=True)
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

try:
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
    logger.info("LOG: STARTUP : Librerías cargadas y SSL_CERT_FILE configurado")
except Exception as e:
    logger.error(f"LOG: FATAL : Error durante la carga de dependencias: {e}", exc_info=True)
    # No salimos todavía para que el health server siga vivo y podamos ver el log
    import time
    time.sleep(300) 
    sys.exit(1)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

atp_win_model = None
wta_win_model = None
player_profiles = pd.DataFrame()

def load_models_and_data():
    global atp_win_model, wta_win_model, player_profiles
    logger.info(f"LOG: IA : Iniciando carga desde {MODELS_DIR}")
    try:
        if not os.path.exists(MODELS_DIR):
            logger.error(f"LOG: IA : Directorio de modelos NO EXISTE: {MODELS_DIR}")
            os.makedirs(MODELS_DIR, exist_ok=True)

        # Modelos de Ganador (Nuevos Pro)
        atp_win_path = os.path.join(MODELS_DIR, 'atp_win_model.pkl')
        wta_win_path = os.path.join(MODELS_DIR, 'wta_win_model.pkl')
        
        logger.info(f"Probando ATP Win en: {atp_win_path}")
        if os.path.exists(atp_win_path):
            atp_win_model = joblib.load(atp_win_path)
            logger.info("Modelo ATP Win Pro cargado con éxito.")
        else:
            logger.warning(f"ADVERTENCIA: Archivo ATP Win NO ENCONTRADO en {atp_win_path}")

        logger.info(f"Probando WTA Win en: {wta_win_path}")
        if os.path.exists(wta_win_path):
            wta_win_model = joblib.load(wta_win_path)
            logger.info("Modelo WTA Win Pro cargado con éxito.")
        else:
            logger.warning(f"ADVERTENCIA: Archivo WTA Win NO ENCONTRADO en {wta_win_path}")
            
        # Perfiles de jugadores
        profiles_path = os.path.join(PROCESSED_DIR, 'player_profiles.csv')
        logger.info(f"Probando perfiles en: {profiles_path}")
        if os.path.exists(profiles_path):
            player_profiles = pd.read_csv(profiles_path)
            logger.info(f"Perfiles de {len(player_profiles)} jugadores cargados.")
        else:
            logger.warning(f"ADVERTENCIA: player_profiles.csv NO ENCONTRADO en {profiles_path}")
            
    except Exception as e:
        logger.error(f"LOG: IA : Error crítico cargando recursos: {e}", exc_info=True)

load_models_and_data()

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
    print("[CRON JOB] Iniciando rutina Pro 24H de automatización MLOps...")
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), 'daily_update.py')
    subprocess.run([python_exe, script_path])
    
    global atp_win_model, wta_win_model, player_profiles
    try:
        # Recargar modelos de Ganador (Pro)
        atp_win_path = os.path.join(MODELS_DIR, 'atp_win_model.pkl')
        wta_win_path = os.path.join(MODELS_DIR, 'wta_win_model.pkl')
        if os.path.exists(atp_win_path): atp_win_model = joblib.load(atp_win_path)
        if os.path.exists(wta_win_path): wta_win_model = joblib.load(wta_win_path)
        
        # Recargar Perfiles
        profiles_path = os.path.join(PROCESSED_DIR, 'player_profiles.csv')
        if os.path.exists(profiles_path): player_profiles = pd.read_csv(profiles_path)
        
        print("[CRON JOB] Modelos Pro y Perfiles re-cargados en caliente con éxito.")
    except Exception as e:
        print(f"[CRON JOB] Error recargando recursos Pro: {e}")
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
    if atp_win_model is None and wta_win_model is None:
        logger.error("Los modelos de IA no están cargados. Abortando scan.")
        await update.message.reply_text("⚠️ ERROR: Los modelos Pro no están cargados. El sistema requiere entrenamiento.")
        return

    responses = ["💰 *Escaneo Pro Bet365 (Próximas 12h)* 💰\n"]
    found_any = False
    
    # Filtrar partidos por tiempo (Próximas 12 horas)
    now = datetime.datetime.now(datetime.timezone.utc)
    twelve_hours_later = now + datetime.timedelta(hours=12)
    
    analyzed_count = 0
    logger.info(f"Procesando {len(matches)} partidos de la API...")
    
    for m in matches:
        commence_time = pd.to_datetime(m['commence_time'])
        if commence_time > twelve_hours_later:
            continue
            
        analyzed_count += 1
        if not m.get('bookmakers'): continue
        bm = m['bookmakers'][0] # bet365
        
        p1_name = m['home_team']
        p2_name = m['away_team']
        sport_key = m['sport_key'] # tennis_atp or tennis_wta
        model = atp_win_model if 'atp' in sport_key else wta_win_model
        
        if model is None: continue

        p1_odds, p2_odds = 0, 0
        for market in bm['markets']:
            if market['key'] == 'h2h':
                for out in market['outcomes']:
                    if out['name'] == p1_name: p1_odds = out['price']
                    else: p2_odds = out['price']
        
        if p1_odds == 0 or p2_odds == 0: continue

        # LOOKUP JUGADORES
        prof1 = player_profiles[player_profiles['name'] == p1_name]
        prof2 = player_profiles[player_profiles['name'] == p2_name]
        
        def get_val(p, col, default):
            return p[col].iloc[0] if not p.empty and not pd.isna(p[col].iloc[0]) else default

        # Extraer características
        rank1, rank2 = get_val(prof1, 'rank', 250), get_val(prof2, 'rank', 250)
        form1, form2 = get_val(prof1, 'form', 0.5), get_val(prof2, 'form', 0.5)
        
        # Detección de superficie por nombre del torneo (simplificado)
        surface = "hard"
        comp_name = m.get('competition_name', '').lower()
        if 'clay' in comp_name or 'tierra' in comp_name or 'roland' in comp_name: surface = "clay"
        elif 'grass' in comp_name or 'hierba' in comp_name or 'wimbledon' in comp_name: surface = "grass"
        
        eff1 = get_val(prof1, f'eff_{surface}', 0.5)
        eff2 = get_val(prof2, f'eff_{surface}', 0.5)
        
        age1, age2 = get_val(prof1, 'age', 26), get_val(prof2, 'age', 26)
        ht1, ht2 = get_val(prof1, 'ht', 185), get_val(prof2, 'ht', 185)
        hand1 = 1 if get_val(prof1, 'hand', 'R') == 'L' else 0
        hand2 = 1 if get_val(prof2, 'hand', 'R') == 'L' else 0
        
        # Dataset para predicción
        df_ml = pd.DataFrame([{
            'rank_diff': rank2 - rank1,
            'age_diff': age1 - age2,
            'ht_diff': ht1 - ht2,
            'form_diff': form1 - form2,
            'surface_diff': eff1 - eff2,
            'p1_rank': rank1, 'p2_rank': rank2,
            'p1_form': form1, 'p2_form': form2,
            'p1_surface_eff': eff1, 'p2_surface_eff': eff2,
            'p1_fatigue': 30, 'p2_fatigue': 30,
            'p1_hand': hand1, 'p2_hand': hand2,
            'tourney_level': 2 # ATP 500 aprox
        }])

        prob_p1_win = model.predict_proba(df_ml)[0][1] # p1_won = 1
        prob_p2_win = 1 - prob_p1_win
        
        texto = f"\n🎾 *{p1_name} vs {p2_name}*\n"
        texto += f"🏆 {m.get('competition_name', 'Torneo')} | 🕒 {commence_time.strftime('%H:%M')}\n"
        bets_found = False
        
        # Valor en P1
        edge_p1 = (prob_p1_win * p1_odds) - 1
        if edge_p1 > 0.05:
            stake = calculate_kelly(prob_p1_win, p1_odds)
            reasons = []
            if eff1 > 0.65: reasons.append(f"Esp. {surface.capitalize()}")
            if form1 > 0.75: reasons.append("Racha Octava")
            if rank1 < rank2 - 60: reasons.append("Mucho Mejor Rank")
            
            texto += f"✅ VALOR: *{p1_name}* @{p1_odds}\n"
            texto += f"   └ Prob: {prob_p1_win*100:.1f}% | Edge: +{edge_p1*100:.1f}%\n"
            if reasons: texto += f"   └ Factores: _{', '.join(reasons)}_\n"
            texto += f"   └ Stake: *{stake:.1f} uds*\n"
            bets_found = True
            
        # Valor en P2
        edge_p2 = (prob_p2_win * p2_odds) - 1
        if edge_p2 > 0.05:
            stake = calculate_kelly(prob_p2_win, p2_odds)
            reasons = []
            if eff2 > 0.65: reasons.append(f"Esp. {surface.capitalize()}")
            if form2 > 0.75: reasons.append("Racha Octava")
            if rank2 < rank1 - 60: reasons.append("Mucho Mejor Rank")
            
            texto += f"✅ VALOR: *{p2_name}* @{p2_odds}\n"
            texto += f"   └ Prob: {prob_p2_win*100:.1f}% | Edge: +{edge_p2*100:.1f}%\n"
            if reasons: texto += f"   └ Factores: _{', '.join(reasons)}_\n"
            texto += f"   └ Stake: *{stake:.1f} uds*\n"
            bets_found = True
            
        if bets_found:
            responses.append(texto)
            found_any = True
            
    if not found_any:
        responses.append(f"\n❌ Analizados {analyzed_count} partidos (12h), pero ninguno ofrece valor suficiente (+5% Edge).")
        
    msg = "\n".join(responses)
    for i in range(0, len(msg), 4000):
        await update.message.reply_text(msg[i:i+4000], parse_mode='Markdown')

async def debug_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra información de diagnóstico del servidor."""
    try:
        text = "🔍 DIAGNÓSTICO DEL SISTEMA 🔍\n\n"
        text += f"📅 Fecha server: {datetime.datetime.now()}\n"
        text += f"📂 Directorio actual: {os.getcwd()}\n"
        text += f"📂 MODELS_DIR: {MODELS_DIR}\n"
        
        # Listar archivos en models
        if os.path.exists(MODELS_DIR):
            files = os.listdir(MODELS_DIR)
            text += f"📄 Archivos en models: {', '.join(files) if files else 'VACÍO'}\n"
        else:
            text += "❌ MODELS_DIR no existe.\n"
            
        text += f"\n🤖 ATP Model: {'Cargado ✅' if atp_win_model else 'FALTA ❌'}\n"
        text += f"🤖 WTA Model: {'Cargado ✅' if wta_win_model else 'FALTA ❌'}\n"
        text += f"📈 Perfiles: {len(player_profiles)} cargados\n"
        
        # No usar Markdown para evitar errores de parseo con caracteres especiales
        await update.message.reply_text(text)
    except Exception as e:
        logger.error(f"Error en debug_info: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error interno en debug: {e}")

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
        application.add_handler(CommandHandler("debug", debug_info))

        t = datetime.time(hour=4, minute=0, tzinfo=datetime.timezone.utc)
        application.job_queue.run_daily(daily_update_job, t)

        logger.info("LOG: Motor API Bot Iniciado. run_polling...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"FATAL: El bot ha crasheado en main(): {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
