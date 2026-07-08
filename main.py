import os
import time
import numpy as np
from stable_baselines3 import PPO
from scorpion_env import ScorpionEnv

def draw_grid(env, info, torque1, torque2):
    # Cuadrícula discreta de texto (50 x 25)
    width, height = 50, 25
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Rango de coordenadas de la simulación
    # El brazo alcanza hasta r=25, x en [-25, 25], y en [-5, 30]
    min_x, max_x = -25.0, 25.0
    min_y, max_y = -5.0, 30.0
    
    def to_grid(x, y):
        col = int((x - min_x) / (max_x - min_x) * (width - 1))
        row = int((max_y - y) / (max_y - min_y) * (height - 1)) # Invertir eje Y para consola
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        return col, row

    # Obtener posiciones de la información devuelta por el entorno
    x_base, y_base = 0.0, 0.0
    x_codo = info.get("x_codo", 0.0)
    y_codo = info.get("y_codo", 0.0)
    x_tip = info.get("x_tip", 0.0)
    y_tip = info.get("y_tip", 0.0)
    
    target_x, target_y = env.target_x, env.target_y

    # Algoritmo sencillo de dibujo de líneas de Bresenham
    def draw_line(x1, y1, x2, y2, char='#'):
        c1, r1 = to_grid(x1, y1)
        c2, r2 = to_grid(x2, y2)
        dc, dr = abs(c2 - c1), abs(r2 - r1)
        sc, sr = 1 if c1 < c2 else -1, 1 if r1 < r2 else -1
        err = dc - dr
        
        while True:
            if grid[r1][c1] == ' ':
                grid[r1][c1] = char
            if c1 == c2 and r1 == r2:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr; c1 += sc
            if e2 < dc:
                err += dc; r1 += sr

    # Dibujar segmentos del brazo
    draw_line(x_base, y_base, x_codo, y_codo, '#')
    draw_line(x_codo, y_codo, x_tip, y_tip, 'o')
    
    # Dibujar articulaciones principales
    cb, rb = to_grid(x_base, y_base)
    cc, rc = to_grid(x_codo, y_codo)
    ct, rt = to_grid(x_tip, y_tip)
    grid[rb][cb] = 'B' # Base
    grid[rc][cc] = 'C' # Codo
    grid[rt][ct] = '*' # Aguijón
    
    # Dibujar objetivo (X representa el objetivo a alcanzar)
    cx, rx = to_grid(target_x, target_y)
    grid[rx][cx] = 'X'
    
    # Renderizar pantalla interactiva
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * (width + 2))
    for row in grid:
        print("|" + "".join(row) + "|")
    print("=" * (width + 2))
    
    # Mostrar estadísticas y torques
    dist = info.get('distance', 0.0)
    print(f"Objetivo: ({target_x:5.1f}, {target_y:5.1f}) | Aguijón: ({x_tip:5.1f}, {y_tip:5.1f})")
    print(f"Distancia al objetivo: {dist:5.2f}")
    if torque1 is not None and torque2 is not None:
        print(f"Torques aplicados => Hombro: {torque1:+.3f} | Codo: {torque2:+.3f}")
    print("Presiona Ctrl+C para salir.")

def main():
    model_file = os.path.join("models", "scorpion_ppo_model.zip")
    if not os.path.exists(model_file):
        print(f"No se encuentra el modelo en '{model_file}'.")
        print("Ejecuta 'python train_scorpion.py' primero para entrenar el modelo.")
        return

    # Cargar política
    print("Cargando modelo...")
    model = PPO.load(model_file, device="cpu") # Para la inferencia y visualización CPU es suficiente
    env = ScorpionEnv()

    try:
        while True:
            obs, _ = env.reset()
            
            # Inicializar primera iteración simulando valores para dibujar el frame 0
            x_codo = env.L1 * np.cos(env.theta1)
            y_codo = env.L1 * np.sin(env.theta1)
            x_tip = x_codo + env.L2 * np.cos(env.theta1 + env.theta2)
            y_tip = y_codo + env.L2 * np.sin(env.theta1 + env.theta2)
            dist = np.sqrt((x_tip - env.target_x)**2 + (y_tip - env.target_y)**2)
            info = {"x_codo": x_codo, "y_codo": y_codo, "x_tip": x_tip, "y_tip": y_tip, "distance": dist}
            
            draw_grid(env, info, 0.0, 0.0)
            time.sleep(0.08)
            
            terminated, truncated = False, False
            while not (terminated or truncated):
                # Usar acciones puramente deterministas de la política
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                t1, t2 = float(action[0]), float(action[1])
                draw_grid(env, info, t1, t2)
                time.sleep(0.08)
                
                if terminated:
                    print("\n¡IMPACTO CRÍTICO! El objetivo ha sido alcanzado (+10 de recompensa).")
                    time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nSimulación ASCII finalizada por el usuario.")

if __name__ == "__main__":
    main()
