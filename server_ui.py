import os
import sys
import math
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
from stable_baselines3 import PPO
from scorpion_env import ScorpionEnv

# Validar y forzar compatibilidad Unicode en consola Windows si es necesario
if sys.platform == "win32":
    import codecs
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

app = FastAPI()

# 1. Intentar cargar el modelo
model_path = os.path.join("models", "scorpion_ppo_model.zip")
model = None
if os.path.exists(model_path):
    print(f"Cargando modelo PPO entrenado desde {model_path}...")
    model = PPO.load(model_path, device="cpu")
else:
    print(f"Advertencia: No se encontró el modelo en {model_path}.")
    print("Se usarán acciones aleatorias como respaldo (stochastic fallback).")

# 2. Interfaz Gráfica HTML5 con Tailwind CSS
html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Scorpion AI - Dashboard Robótico</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
        #canvas-container { display: flex; justify-content: center; align-items: center; }
        canvas { background-color: #1e293b; border: 2px solid #334155; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
        .overlay { display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(15, 23, 42, 0.85); justify-content: center; align-items: center; flex-direction: column; z-index: 50; }
        .overlay.active { display: flex; }
        .log-console { height: 160px; overflow-y: auto; background-color: #000; font-family: monospace; font-size: 0.9rem;}
        .log-console span.info { color: #3b82f6; }
        .log-console span.success { color: #22c55e; }
        .log-console span.warning { color: #f59e0b; }
    </style>
</head>
<body class="min-h-screen p-6 relative overflow-hidden">
    
    <!-- Pantalla de Fin de Partida -->
    <div id="end-overlay" class="overlay rounded-xl">
        <h2 id="end-title" class="text-6xl font-black mb-4 tracking-tight"></h2>
        <p id="end-subtitle" class="text-2xl text-slate-300 mb-8 font-light"></p>
        <button onclick="location.reload()" class="px-8 py-4 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl shadow-lg transition-all transform hover:scale-105">
            Lanzar Nuevo Episodio
        </button>
    </div>

    <header class="mb-8 text-center">
        <h1 class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-400">
            Escorpión Luchador: Telemetría PPO
        </h1>
        <p class="text-slate-400 mt-2 font-medium">Control Cinemático Continuo en Tiempo Real</p>
    </header>

    <div class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6 relative">
        
        <!-- Panel Izquierdo: Telemetría -->
        <div class="col-span-1 flex flex-col gap-4">
            <!-- Ángulos -->
            <div class="bg-slate-800 p-5 rounded-2xl shadow-xl border border-slate-700">
                <h3 class="text-lg font-semibold text-slate-200 mb-3 border-b border-slate-600 pb-2 flex items-center gap-2">
                    <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                    Articulaciones
                </h3>
                <div class="flex justify-between mb-2">
                    <span class="text-slate-400">Hombro (θ₁)</span>
                    <span id="t-theta1" class="font-mono text-cyan-300 text-lg">0.0°</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-slate-400">Codo (θ₂)</span>
                    <span id="t-theta2" class="font-mono text-blue-400 text-lg">0.0°</span>
                </div>
            </div>

            <!-- Torques -->
            <div class="bg-slate-800 p-5 rounded-2xl shadow-xl border border-slate-700">
                <h3 class="text-lg font-semibold text-slate-200 mb-3 border-b border-slate-600 pb-2 flex items-center gap-2">
                    <svg class="w-5 h-5 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                    Fuerza Actuadores (Torque)
                </h3>
                <div class="mb-5">
                    <div class="flex justify-between text-sm mb-1">
                        <span class="text-slate-400">Hombro</span>
                        <span id="t-torque1" class="font-mono text-cyan-400 font-bold">0.00</span>
                    </div>
                    <div class="w-full bg-slate-900 rounded-full h-3 relative overflow-hidden">
                        <div id="bar-torque1" class="bg-cyan-500 h-3 rounded-full absolute top-0" style="width: 0%; left: 50%;"></div>
                        <div class="absolute w-0.5 h-3 bg-slate-600 left-1/2 top-0"></div>
                    </div>
                </div>
                <div>
                    <div class="flex justify-between text-sm mb-1">
                        <span class="text-slate-400">Codo</span>
                        <span id="t-torque2" class="font-mono text-blue-500 font-bold">0.00</span>
                    </div>
                    <div class="w-full bg-slate-900 rounded-full h-3 relative overflow-hidden">
                        <div id="bar-torque2" class="bg-blue-600 h-3 rounded-full absolute top-0" style="width: 0%; left: 50%;"></div>
                        <div class="absolute w-0.5 h-3 bg-slate-600 left-1/2 top-0"></div>
                    </div>
                </div>
            </div>

            <!-- Objetivo -->
            <div class="bg-slate-800 p-5 rounded-2xl shadow-xl border border-slate-700">
                <h3 class="text-lg font-semibold text-slate-200 mb-3 border-b border-slate-600 pb-2">Tracking Espacial</h3>
                <div class="flex justify-between mb-2 items-center">
                    <span class="text-slate-400">Distancia Euclidiana</span>
                    <span id="t-dist" class="font-mono text-rose-400 text-2xl font-bold tracking-tight">0.00</span>
                </div>
            </div>
        </div>

        <!-- Panel Central: Canvas y Logs -->
        <div class="col-span-1 lg:col-span-2 relative flex flex-col gap-4">
            <div id="canvas-container" class="w-full relative">
                <!-- Se dibuja el brazo robótico -->
                <canvas id="simCanvas" width="800" height="450" class="w-full"></canvas>
            </div>
            
            <div class="bg-black/80 rounded-xl border border-slate-700 p-4 log-console shadow-inner" id="log-console">
                <span class="text-slate-500 block mb-2">--- Registro de Eventos ---</span>
            </div>
        </div>
    </div>

    <script>
        // 3. Lógica del Frontend
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const W = canvas.width;
        const H = canvas.height;

        // Geometría y Escala
        // El brazo estirado tiene L1+L2 = 25 unidades.
        // Haremos que r=30 ocupe la mayor parte de la altura del canvas.
        const SCALE = 10.0; 
        const OFFSET_X = W / 2;       // Base centrada horizontalmente
        const OFFSET_Y = H - 50;      // Base cerca del suelo del canvas

        function toPx(x, y) {
            return {
                cx: OFFSET_X + x * SCALE,
                cy: OFFSET_Y - y * SCALE // Invertir eje Y para que apunte hacia arriba
            };
        }

        function drawFrame(data) {
            // Limpiar fondo
            ctx.clearRect(0, 0, W, H);
            
            // Dibujar rejilla (grid) de referencia
            ctx.strokeStyle = '#334155';
            ctx.lineWidth = 1;
            for(let i = 0; i <= W; i += 50) { 
                ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, H); ctx.stroke(); 
            }
            for(let j = 0; j <= H; j += 50) { 
                ctx.beginPath(); ctx.moveTo(0, j); ctx.lineTo(W, j); ctx.stroke(); 
            }

            // Suelo
            ctx.strokeStyle = '#475569';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(0, OFFSET_Y);
            ctx.lineTo(W, OFFSET_Y);
            ctx.stroke();

            // Transformar coordenadas físicas a píxeles
            const pBase = toPx(0, 0);
            const pCodo = toPx(data.x_codo, data.y_codo);
            const pTip = toPx(data.x_tip, data.y_tip);
            const pTarget = toPx(data.target_x, data.target_y);

            // 3.1 Dibujar Objetivo (Diana concéntrica roja)
            ctx.beginPath();
            ctx.arc(pTarget.cx, pTarget.cy, 18, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(225, 29, 72, 0.15)';
            ctx.fill();
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#e11d48';
            ctx.stroke();
            
            ctx.beginPath();
            ctx.arc(pTarget.cx, pTarget.cy, 8, 0, 2 * Math.PI);
            ctx.fillStyle = '#e11d48';
            ctx.fill();
            
            // Cruz en el objetivo
            ctx.beginPath();
            ctx.moveTo(pTarget.cx - 24, pTarget.cy); ctx.lineTo(pTarget.cx + 24, pTarget.cy);
            ctx.moveTo(pTarget.cx, pTarget.cy - 24); ctx.lineTo(pTarget.cx, pTarget.cy + 24);
            ctx.strokeStyle = 'rgba(225, 29, 72, 0.5)';
            ctx.stroke();

            // 3.2 Dibujar Hombro (Segmento 1)
            ctx.beginPath();
            ctx.moveTo(pBase.cx, pBase.cy);
            ctx.lineTo(pCodo.cx, pCodo.cy);
            ctx.lineWidth = 14;
            ctx.lineCap = 'round';
            ctx.strokeStyle = '#0ea5e9'; // Cian vibrante
            ctx.stroke();

            // 3.3 Dibujar Codo (Segmento 2)
            ctx.beginPath();
            ctx.moveTo(pCodo.cx, pCodo.cy);
            ctx.lineTo(pTip.cx, pTip.cy);
            ctx.lineWidth = 10;
            ctx.strokeStyle = '#3b82f6'; // Azul cobalto
            ctx.stroke();

            // 3.4 Dibujar Articulaciones (Nodos)
            // Base
            ctx.beginPath(); ctx.arc(pBase.cx, pBase.cy, 12, 0, 2*Math.PI); 
            ctx.fillStyle = '#cbd5e1'; ctx.fill(); ctx.stroke();
            // Codo central
            ctx.beginPath(); ctx.arc(pCodo.cx, pCodo.cy, 9, 0, 2*Math.PI); 
            ctx.fillStyle = '#f8fafc'; ctx.fill();
            // Aguijón (Tip)
            ctx.beginPath(); ctx.arc(pTip.cx, pTip.cy, 6, 0, 2*Math.PI); 
            ctx.fillStyle = '#fde047'; ctx.fill(); // Amarillo brillante
        }

        function updateTelemetry(data) {
            // Conversión de radianes a grados
            const t1_deg = (data.theta1 * 180 / Math.PI).toFixed(1);
            const t2_deg = (data.theta2 * 180 / Math.PI).toFixed(1);
            document.getElementById('t-theta1').textContent = t1_deg + '°';
            document.getElementById('t-theta2').textContent = t2_deg + '°';

            // Actualización de barras de torque
            document.getElementById('t-torque1').textContent = data.torque1.toFixed(3);
            document.getElementById('t-torque2').textContent = data.torque2.toFixed(3);
            
            // Torque va de -1.0 a 1.0. 
            // Representamos magnitud con el ancho (0 a 50%) y dirección con el offset.
            const t1_pct = Math.abs(data.torque1) * 50;
            const b1 = document.getElementById('bar-torque1');
            b1.style.width = t1_pct + '%';
            b1.style.left = data.torque1 < 0 ? (50 - t1_pct) + '%' : '50%';

            const t2_pct = Math.abs(data.torque2) * 50;
            const b2 = document.getElementById('bar-torque2');
            b2.style.width = t2_pct + '%';
            b2.style.left = data.torque2 < 0 ? (50 - t2_pct) + '%' : '50%';

            // Distancia en tiempo real
            document.getElementById('t-dist').textContent = data.distance.toFixed(3);
        }

        function logConsole(msg, type="info") {
            const consoleDiv = document.getElementById('log-console');
            const span = document.createElement('span');
            span.className = type + ' block';
            span.textContent = `> ${msg}`;
            consoleDiv.appendChild(span);
            consoleDiv.scrollTop = consoleDiv.scrollHeight;
        }

        // 4. Conexión WebSocket Asíncrona
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            logConsole("Conexión WebSocket establecida con el servidor de IA.", "success");
            logConsole("Iniciando episodio continuo...", "info");
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.status === "running" || data.status === "ended") {
                drawFrame(data);
                updateTelemetry(data);
                
                // Reducir spam de logs, imprimir 1 vez por segundo (cada 20 frames aprox a 0.05s)
                if (data.status === "running" && data.step % 15 === 0) {
                    logConsole(`[Frame ${data.step}] Ajustando torques. Recompensa actual: ${data.reward.toFixed(2)}`);
                }

                if (data.status === "ended") {
                    const overlay = document.getElementById('end-overlay');
                    const title = document.getElementById('end-title');
                    const subtitle = document.getElementById('end-subtitle');
                    
                    overlay.classList.add('active');
                    
                    if (data.reason === "success") {
                        title.textContent = "¡Impacto Exitoso!";
                        title.className = "text-6xl font-black mb-4 tracking-tight text-emerald-400";
                        subtitle.textContent = `El agente alcanzó el objetivo en ${data.step} pasos.`;
                        logConsole("¡OBJETIVO ABATIDO CON ÉXITO! Bono de Impacto otorgado.", "success");
                    } else {
                        title.textContent = "Límite de Tiempo Alcanzado";
                        title.className = "text-6xl font-black mb-4 tracking-tight text-rose-500";
                        subtitle.textContent = "El escorpión no logró llegar a la diana a tiempo.";
                        logConsole("Simulación terminada por truncamiento (max_steps alcanzado).", "warning");
                    }
                    ws.close();
                }
            }
        };

        ws.onclose = () => {
            logConsole("Enlace de telemetría cerrado.", "warning");
        };
    </script>
</body>
</html>
"""

# 5. Endpoints de FastAPI
@app.get("/")
async def serve_ui():
    """Sirve la interfaz web estática de Tailwind + Canvas."""
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enlace de telemetría bidireccional en tiempo real."""
    await websocket.accept()
    env = ScorpionEnv()
    obs, info = env.reset()
    
    step = 0
    try:
        while True:
            # Seleccionar acción con PPO o modo aleatorio de respaldo
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            # Ejecutar el paso físico
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Torques limpios
            t1, t2 = float(action[0]), float(action[1])
            
            # Empaquetar estado para JSON (todo debe ser nativo float/int)
            payload = {
                "status": "running",
                "step": step,
                "theta1": float(env.theta1),
                "theta2": float(env.theta2),
                "torque1": t1,
                "torque2": t2,
                "x_codo": float(info["x_codo"]),
                "y_codo": float(info["y_codo"]),
                "x_tip": float(info["x_tip"]),
                "y_tip": float(info["y_tip"]),
                "target_x": float(env.target_x),
                "target_y": float(env.target_y),
                "distance": float(info["distance"]),
                "reward": float(reward)
            }
            
            # Evaluar fin de episodio
            if terminated or truncated:
                payload["status"] = "ended"
                payload["reason"] = "success" if terminated else "timeout"
                await websocket.send_text(json.dumps(payload))
                break
            
            # Transmitir frame
            await websocket.send_text(json.dumps(payload))
            
            # Retardo asíncrono para suavizar la renderización física (50 ms por frame)
            await asyncio.sleep(0.05)
            
    except WebSocketDisconnect:
        print("Cliente WebSocket desconectado desde el navegador.")
    finally:
        env.close()

if __name__ == "__main__":
    import uvicorn
    # Inicializar el servidor Uvicorn en el puerto 8001 para evitar conflictos
    uvicorn.run(app, host="127.0.0.1", port=8001)
