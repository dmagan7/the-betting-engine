import sys, os, asyncio
print("LOG: BOOT : Bot script started at the very first line.", flush=True)
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# 1. Configuracion de logging INMEDIATA
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# 2. Servidor de Salud ULTRA-RAPIDO para Azure (Antes que pandas/numpy)
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
        logger.error(f"LOG: ERROR : Fallo critico en Servidor de Salud: {e}")

# LANZAR SALUD YA MISMO (Hilo no-bloqueante)
logger.info("LOG: STARTUP : Iniciando health check server...")
threading.Thread(target=run_health_server, daemon=True).start()

logger.info("LOG: STARTUP : Cargando librerias pesadas (pandas, sklearn, etc)...")

try:
    import joblib
    import pandas as pd
    import numpy as np
    import datetime
    import subprocess
    import certifi
    import requests
    import httpx
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    from dotenv import load_dotenv
    from scipy.stats import poisson

    load_dotenv()
    os.environ['SSL_CERT_FILE'] = certifi.where()
    logger.info("LOG: STARTUP : Librerias cargadas y SSL_CERT_FILE configurado")
except Exception as e:
    logger.error(f"LOG: FATAL : Error durante la carga de dependencias: {e}", exc_info=True)
    # No salimos todavia para que el health server siga vivo y podamos ver el log
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
            logger.info("Modelo ATP Win Pro cargado con exito.")
        else:
            logger.warning(f"ADVERTENCIA: Archivo ATP Win NO ENCONTRADO en {atp_win_path}")

        logger.info(f"Probando WTA Win en: {wta_win_path}")
        if os.path.exists(wta_win_path):
            wta_win_model = joblib.load(wta_win_path)
            logger.info("Modelo WTA Win Pro cargado con exito.")
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
        logger.error(f"LOG: IA : Error critico cargando recursos: {e}", exc_info=True)

load_models_and_data()

def calculate_kelly(prob, odds, fraction=0.25, max_stake=5.0):
    if prob == 0 or odds <= 1: return 0.0
    q = 1 - prob
    b = odds - 1
    kelly_pct = ((b * prob) - q) / b
    if kelly_pct <= 0: return 0.0
    return min(kelly_pct * fraction * 100, max_stake)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("The Betting Engine Live. Usa /valuebets para escanear Bet365 en tiempo real.")

async def daily_update_job(context: ContextTypes.DEFAULT_TYPE):
    print("--------------------------------------------------")
    print("[CRON JOB] Iniciando rutina Pro 24H de automatizacion MLOps...")
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
        
        print("[CRON JOB] Modelos Pro y Perfiles re-cargados en caliente con exito.")
    except Exception as e:
        print(f"[CRON JOB] Error recargando recursos Pro: {e}")
    print("--------------------------------------------------")

async def valuebets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    progress_msg = await update.message.reply_text("[SEARCH] Iniciando escaneo en tiempo real de The Odds API...")
    ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "547f9d4f748137ff6adbf7fe48baa1a7")
    
    async def update_status(text):
        try:
            await progress_msg.edit_text(f"[SEARCH] {text}")
        except Exception:
            pass # Ignorar errores si el mensaje no puede ser editado (ej. mismo texto)

    # 1. Obtener los deportes activos
    await update_status("Obteniendo deportes de tenis activos...")
    sports_url = "https://api.the-odds-api.com/v4/sports"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(sports_url, params={"apiKey": ODDS_API_KEY})
            resp.raise_for_status()
            sports_data = resp.json()
        
        tennis_sports = [s['key'] for s in sports_data if 'tennis' in s['key'].lower()]
        logger.info(f"Deportes de tenis encontrados: {len(tennis_sports)}")
    except Exception as e:
        logger.error(f"Error conectando a The Odds API (Sports): {e}")
        await update_status(f"[FAIL] Error conectando servidor API para deportes: {e}")
        return

    if not tennis_sports:
        await update_status("[FAIL] No hay torneos de tenis activos en The Odds API actualmente.")
        return

    matches = []
    # 2. Descargar cuotas para cada torneo de tenis activo
    for i, sport_key in enumerate(tennis_sports):
        await update_status(f"Descargando cuotas: {sport_key} ({i+1}/{len(tennis_sports)})...")
        odds_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h"}
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(odds_url, params=params)
                resp.raise_for_status()
                sport_matches = resp.json()
                if isinstance(sport_matches, list):
                    matches.extend(sport_matches)
        except Exception as e:
            logger.warning(f"Error descargando cuotas para {sport_key}: {e}")
            continue

    if not matches:
        await update_status("[FAIL] No hay partidos en la base de datos oficial de The Odds API actualmente.")
        return

    await update_status(f"Analizando {len(matches)} partidos con modelos de IA Pro...")

    # Verificacion de modelos antes de procesar
    if atp_win_model is None and wta_win_model is None:
        logger.error("Los modelos de IA no estan cargados. Abortando scan.")
        await update.message.reply_text("[WARN] ERROR: Los modelos Pro no estan cargados. El sistema requiere entrenamiento.")
        return

    responses = ["[$$] *Escaneo Pro - Mejores Cuotas EU (Proximas 24h)* [$$]\n"]
    found_any = False
    
    # Filtrar partidos por tiempo (Proximas 24 horas)
    now = datetime.datetime.now(datetime.timezone.utc)
    twentyfour_hours_later = now + datetime.timedelta(hours=24)
    
    analyzed_count = 0
    analyzed_log = []
    skipped_time = 0
    logger.info(f"Procesando {len(matches)} partidos de la API...")
    
    skipped_no_bm = 0
    skipped_no_model = 0
    skipped_no_odds = 0
    skip_details = []
    
    for m in matches:
        # BUG FIX 1: pd.to_datetime sin utc=True produce timestamps tz-naive
        # que no se pueden comparar con now (tz-aware), saltandose todos los partidos silenciosamente
        commence_time = pd.to_datetime(m['commence_time'], utc=True)
        if commence_time > twentyfour_hours_later:
            skipped_time += 1
            continue
            
        analyzed_count += 1
        bookmakers_list = m.get('bookmakers', [])
        if not bookmakers_list:
            skipped_no_bm += 1
            skip_details.append(f"  * {m.get('home_team','?')} vs {m.get('away_team','?')} - sin cuotas en ninguna casa EU")
            continue

        # Preferir Bet365, si no la primera disponible
        bm = next((b for b in bookmakers_list if b['key'] == 'bet365'), bookmakers_list[0])
        bm_name = bm.get('title', bm['key'])
        
        p1_name = m['home_team']
        p2_name = m['away_team']
        sport_key = m['sport_key'] # tennis_atp or tennis_wta
        model = atp_win_model if 'atp' in sport_key else wta_win_model
        
        if model is None:
            skipped_no_model += 1
            skip_details.append(f"  * {p1_name} vs {p2_name} - modelo {sport_key} no entrenado")
            logger.warning(f"Modelo None para sport_key={sport_key}, saltando {p1_name} vs {p2_name}")
            continue

        p1_odds, p2_odds = 0, 0
        for market in bm['markets']:
            if market['key'] == 'h2h':
                for out in market['outcomes']:
                    if out['name'] == p1_name: p1_odds = out['price']
                    else: p2_odds = out['price']
        
        if p1_odds == 0 or p2_odds == 0:
            skipped_no_odds += 1
            skip_details.append(f"  * {p1_name} vs {p2_name} - sin cuota H2H en {bm_name}")
            continue

        # LOOKUP JUGADORES
        prof1 = player_profiles[player_profiles['name'] == p1_name]
        prof2 = player_profiles[player_profiles['name'] == p2_name]
        
        p1_found = not prof1.empty
        p2_found = not prof2.empty
        
        def get_val(p, col, default):
            if p.empty: return default
            val = p[col].iloc[0]
            return default if pd.isna(val) else val

        # Probabilidad implicita del bookmaker (sin margen) como prior
        # BUG FIX 2: Cuando los jugadores no estan en perfiles, los defaults identicos
        # hacen que el modelo prediga exactamente 50% para ambos -> nunca hay edge.
        # Usamos la prob implicita del bookmaker como punto de partida mas realista.
        total_implied = (1/p1_odds) + (1/p2_odds)
        bm_prob_p1 = (1/p1_odds) / total_implied  # prob normalizada sin margen
        bm_prob_p2 = (1/p2_odds) / total_implied

        # Extraer caracteristicas
        rank1 = get_val(prof1, 'rank', None)
        rank2 = get_val(prof2, 'rank', None)
        
        # Si falta el ranking, inferirlo desde las cuotas del bookmaker
        # Un favorito con cuota 1.5 implica aprox ranking relativo mejor
        if rank1 is None and rank2 is None:
            rank1, rank2 = 100, 100  # sin diferencia
        elif rank1 is None:
            rank1 = int(rank2 * (p1_odds / p2_odds))  # estimacion proporcional
        elif rank2 is None:
            rank2 = int(rank1 * (p2_odds / p1_odds))
            
        form1 = get_val(prof1, 'form', bm_prob_p1)  # si no hay datos, usar prob implicita
        form2 = get_val(prof2, 'form', bm_prob_p2)
        
        # Deteccion de superficie por nombre del torneo (simplificado)
        surface = "hard"
        sport_title = m.get('sport_title', '').lower()
        comp_name = m.get('competition_name', m.get('sport_title', '')).lower()
        if 'clay' in comp_name or 'tierra' in comp_name or 'roland' in comp_name or 'french' in comp_name: surface = "clay"
        elif 'grass' in comp_name or 'hierba' in comp_name or 'wimbledon' in comp_name: surface = "grass"
        
        eff1 = get_val(prof1, f'eff_{surface}', bm_prob_p1)
        eff2 = get_val(prof2, f'eff_{surface}', bm_prob_p2)
        
        age1, age2 = get_val(prof1, 'age', 26), get_val(prof2, 'age', 26)
        ht1, ht2 = get_val(prof1, 'ht', 185), get_val(prof2, 'ht', 185)
        hand1 = 1 if get_val(prof1, 'hand', 'R') == 'L' else 0
        hand2 = 1 if get_val(prof2, 'hand', 'R') == 'L' else 0
        
        # Dataset para prediccion
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

        ml_prob_p1 = model.predict_proba(df_ml)[0][1]  # p1_won = 1
        ml_prob_p2 = 1 - ml_prob_p1
        
        # Blend entre ML y bookmaker segun disponibilidad de perfil
        if p1_found and p2_found:
            alpha = 0.80
        elif p1_found or p2_found:
            alpha = 0.60
        else:
            alpha = 0.40
        
        # SANITY CHECK: si el modelo diverge >2.5x de lo que implica el bookmaker
        # (ej: nosotros damos 14% a alguien que Pinnacle pone al 5%), reducimos alpha.
        # Pinnacle es la casa mas afilada del mundo - si discrepamos tanto, probablemente
        # es un error del modelo (ranking/perfil desactualizado), no valor real.
        ratio_p1 = ml_prob_p1 / bm_prob_p1 if bm_prob_p1 > 0 else 1
        ratio_p2 = ml_prob_p2 / bm_prob_p2 if bm_prob_p2 > 0 else 1
        if max(ratio_p1, ratio_p2) > 2.5:
            alpha = min(alpha, 0.35)  # cortar confianza en ML si diverge demasiado
        
        prob_p1_win = alpha * ml_prob_p1 + (1 - alpha) * bm_prob_p1
        prob_p2_win = alpha * ml_prob_p2 + (1 - alpha) * bm_prob_p2
        
        logger.info(f"  {p1_name} vs {p2_name} | perfiles: {p1_found}/{p2_found} | ML: {ml_prob_p1:.2%} | BM: {bm_prob_p1:.2%} | Final: {prob_p1_win:.2%} | ratio: {max(ratio_p1,ratio_p2):.1f}x | alpha: {alpha}")

        
        texto = f"\n\U0001f3be *{p1_name} vs {p2_name}*\n"
        texto += f"\U0001f3c6 {m.get('sport_title', 'Torneo')} | \U0001f552 {commence_time.strftime('%H:%M')} | \U0001f4cc {bm_name}\n"
        bets_found = False
        
        # Umbral dinamico: cuanto mayor la cuota, mayor la exigencia de edge.
        # Con cuotas altas el modelo comete mas errores de calibracion, por eso
        # pedimos mas margen de seguridad. Valores empiricos de apuestas profesionales:
        #  - Favorito (cuota <=2): basta con 3% de edge real
        #  - Normal (cuota 2-5): exigimos 5%
        #  - Largo plazo (cuota >5): exigimos 8% - cada punto de prob vale mucho
        def min_edge(odds):
            if odds <= 2.0: return 0.03
            elif odds <= 5.0: return 0.05
            else: return 0.08
        
        # Valor en P1
        edge_p1 = (prob_p1_win * p1_odds) - 1
        if edge_p1 > min_edge(p1_odds):
            stake = calculate_kelly(prob_p1_win, p1_odds)
            reasons = []
            if eff1 > 0.65: reasons.append(f"Esp. {surface.capitalize()}")
            if form1 > 0.75: reasons.append("Racha")
            if rank1 < rank2 - 60: reasons.append("Mejor Rank")
            edge_display = min(edge_p1 * 100, 50.0)  # cap visual a 50% - >50% = modelo poco fiable
            
            texto += f"\u2705 VALOR: *{p1_name}* @{p1_odds}\n"
            texto += f"   \u2514 Prob: {prob_p1_win*100:.1f}% | Edge: +{edge_display:.1f}%{'\u26a0\ufe0f' if edge_p1>0.5 else ''}\n"
            if reasons: texto += f"   \u2514 Factores: _{', '.join(reasons)}_\n"
            texto += f"   \u2514 Stake: *{stake:.1f} uds*\n"
            bets_found = True
            
        # Valor en P2
        edge_p2 = (prob_p2_win * p2_odds) - 1
        if edge_p2 > min_edge(p2_odds):
            stake = calculate_kelly(prob_p2_win, p2_odds)
            reasons = []
            if eff2 > 0.65: reasons.append(f"Esp. {surface.capitalize()}")
            if form2 > 0.75: reasons.append("Racha")
            if rank2 < rank1 - 60: reasons.append("Mejor Rank")
            edge_display = min(edge_p2 * 100, 50.0)
            
            texto += f"\u2705 VALOR: *{p2_name}* @{p2_odds}\n"
            texto += f"   \u2514 Prob: {prob_p2_win*100:.1f}% | Edge: +{edge_display:.1f}%{'\u26a0\ufe0f' if edge_p2>0.5 else ''}\n"
            if reasons: texto += f"   \u2514 Factores: _{', '.join(reasons)}_\n"
            texto += f"   \u2514 Stake: *{stake:.1f} uds*\n"
            bets_found = True
            
        if bets_found:
            responses.append(texto)
            found_any = True
        else:
            # Registrar para el resumen
            max_edge = max(edge_p1, edge_p2) * 100
            fav = p1_name if edge_p1 > edge_p2 else p2_name
            analyzed_log.append(f"* {p1_name} v {p2_name} - edge: {max_edge:.1f}% (favor: {fav})")
            
    if not found_any:
        responses.append(f"\n[FAIL] Ninguno de los {analyzed_count} partidos analizados ofrece valor suficiente (+3% Edge).\n")
    
    # Bloque de diagnostico siempre visible
    diag = f"\n[STATS] *Diagnostico*: {len(matches)} en API | {skipped_time} fuera-24h | {analyzed_count} en ventana\n"
    diag += f"  |_ Sin cuotas Bet365: {skipped_no_bm} | Sin modelo: {skipped_no_model} | Sin H2H: {skipped_no_odds}\n"
    diag += f"  |_ Con edge calculado: {len(analyzed_log)}"
    responses.append(diag)
    
    if skip_details:
        responses.append("\n[SEARCH] *Detalle de partidos saltados*:")
        for d in skip_details[:10]:
            responses.append(d)
        if len(skip_details) > 10:
            responses.append(f"...y {len(skip_details) - 10} mas.")        
    if analyzed_log:
        responses.append("\n[NOTES] *Analisis detallado (Top 15)*:")
        # Mostrar los 15 con mayor edge para no saturar Telegram
        analyzed_log_sorted = sorted(analyzed_log, key=lambda x: float(x.split('edge: ')[1].split('%')[0]) if 'edge: ' in x else -999, reverse=True)
        for log_line in analyzed_log_sorted[:15]:
            responses.append(log_line)
        if len(analyzed_log) > 15:
            responses.append(f"...y {len(analyzed_log) - 15} mas.")
    msg = "\n".join(responses)
    for i in range(0, len(msg), 4000):
        await update.message.reply_text(msg[i:i+4000], parse_mode='Markdown')

async def debug_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra informacion de diagnostico del servidor."""
    try:
        text = "[SEARCH] DIAGNOSTICO DEL SISTEMA [SEARCH]\n\n"
        text += f"[DATE] Fecha server: {datetime.datetime.now()}\n"
        text += f"[DIR] Directorio actual: {os.getcwd()}\n"
        text += f"[DIR] MODELS_DIR: {MODELS_DIR}\n"
        
        # Listar archivos en models
        if os.path.exists(MODELS_DIR):
            files = os.listdir(MODELS_DIR)
            text += f"[FILE] Archivos en models: {', '.join(files) if files else 'VACIO'}\n"
        else:
            text += "[FAIL] MODELS_DIR no existe.\n"
            
        text += f"\n[AI] ATP Model: {'Cargado [OK]' if atp_win_model else 'FALTA [FAIL]'}\n"
        text += f"[AI] WTA Model: {'Cargado [OK]' if wta_win_model else 'FALTA [FAIL]'}\n"
        text += f"[UP] Perfiles: {len(player_profiles)} cargados\n"
        
        # No usar Markdown para evitar errores de parseo con caracteres especiales
        await update.message.reply_text(text)
    except Exception as e:
        logger.error(f"Error en debug_info: {e}", exc_info=True)
        await update.message.reply_text(f"[FAIL] Error interno en debug: {e}")

is_training = False

async def train_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_training
    if is_training:
        await update.message.reply_text("[WARN] El proceso de entrenamiento ya esta en curso. Por favor, espera.")
        return
        
    await update.message.reply_text("[WAIT] Iniciando entrenamiento y automatizacion MLOps en segundo plano. Esto tardara unos minutos...")
    
    async def run_training_bg():
        global is_training
        is_training = True
        try:
            logger.info("LOG: MLOPS : Iniciando entrenamiento forzado desde Telegram...")
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), 'daily_update.py')
            
            # Ejecutar de forma asincrona capturando salida para depuracion
            process = await asyncio.create_subprocess_exec(
                python_exe, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            # Decodificar logs (capturamos ambos por si falla el decode de uno)
            full_log = stdout.decode('utf-8', errors='replace')
            error_log = stderr.decode('utf-8', errors='replace')
            combined_log = f"{full_log}\n\n--- ERRORES / STDERR ---\n{error_log}"
            
            if process.returncode == 0:
                logger.info("LOG: MLOPS : Recargando modelos y perfiles...")
                load_models_and_data()
                await update.message.reply_text("[OK] *ENTRENAMIENTO COMPLETADO*\nModelos y perfiles generados.", parse_mode='Markdown')
            else:
                await update.message.reply_text(f"[WARN] *ERROR DURANTE EL ENTRENAMIENTO* (Exit code: {process.returncode})", parse_mode='Markdown')
            
            # Enviar logs en fragmentos (max 4000 caracteres por mensaje de Telegram)
            await update.message.reply_text("--- DETALLE DE LOGS DE ENTRENAMIENTO ---")
            for i in range(0, len(combined_log), 4000):
                chunk = combined_log[i:i+4000]
                # Escapar markdown problemático en logs si es necesario, o usar bloque de codigo
                await update.message.reply_text(f"```\n{chunk}\n```", parse_mode='Markdown')

        except Exception as e:
            logger.error(f"LOG: ERROR : Fallo en el entrenamiento: {e}")
            await update.message.reply_text(f"[FAIL] Error critico en entrenamiento: {e}")
        finally:
            is_training = False

    # Ejecutar tarea asincrona en segundo plano nativamente
    import asyncio
    asyncio.create_task(run_training_bg())

def main():
    try:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if token:
            token = token.strip('"').strip("'").strip()
        
        if not token or token.startswith("tu_token"):
            logger.error("LOG: FATAL : TELEGRAM_BOT_TOKEN no configurado en Azure App Settings.")
            logger.error("LOG: FATAL : El contenedor se cerrara ahora para evitar bucles de error.")
            return
        
        logger.info(f"LOG: STARTUP : Token validado (hash: {hash(token)})")
            
        application = Application.builder().token(token).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("valuebets", valuebets))
        application.add_handler(CommandHandler("debug", debug_info))
        application.add_handler(CommandHandler("train", train_models))

        t = datetime.time(hour=4, minute=0, tzinfo=datetime.timezone.utc)
        application.job_queue.run_daily(daily_update_job, t)

        logger.info("LOG: Motor API Bot Iniciado. run_polling...")
        
        # Bucle para manejar conflictos de polling en Azure (Blue/Green deploys)
        import time
        from telegram.error import Conflict
        
        while True:
            try:
                application.run_polling(allowed_updates=Update.ALL_TYPES)
                break  # Si termina limpio (ej. senal del sistema), salimos
            except Conflict:
                logger.warning("LOG: CONFLICT : Otra instancia de polling activa (posible solapamiento de Azure). Reintentando en 15s...")
                time.sleep(15)
            except Exception as e:
                logger.error(f"LOG: ERROR : Error inesperado en polling: {e}. Reintentando en 20s...")
                time.sleep(20)
                
    except Exception as e:
        logger.critical(f"FATAL: Error irrecuperable en main(): {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
