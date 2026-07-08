# Megaprompt de Construcción: Experimento 05 - El Escorpión Luchador (PPO Continuo con CUDA)

Actúa como un **Ingeniero de Machine Learning Senior experto en Control de Sistemas Robóticos y Cinemática Inversa**. Genera la estructura de código limpia, modular, documentada y 100% funcional para el **Experimento 05: El Escorpión Luchador**, el cual debe ubicarse dentro de la ruta `Aprendizaje_Por_Refuerzo/05_escorpion/`.

El sistema debe programarse en **Python 3** utilizando **Gymnasium** para el entorno de física continua y **Stable-Baselines3 (SB3)** con **PyTorch** para el algoritmo **PPO (Proximal Policy Optimization)**, forzando la ejecución en hardware mediante `device="cuda"`. La simulación física cinemática del péndulo doble debe renderizarse cuadro por cuadro de forma interactiva en la consola con arte ASCII.

---

## 1. Especificaciones Técnicas del Brazo Articulado

El escorpión consta de una base fija en la posición (0,0) de un plano cartesiano, un primer segmento (Hombro de longitud L₁ = 15.0) y un segundo segmento (Codo de longitud L₂ = 10.0). 

*   **Cinemática Directa:** La posición del "codo" es:
    \[X_{codo} = L_1 \cdot \cos(\theta_1), \quad Y_{codo} = L_1 \cdot \sin(\theta_1)\]
    La posición del "aguijón" (extremo final) es:
    \[X_{tip} = X_{codo} + L_2 \cdot \cos(\theta_1 + \theta_2), \quad Y_{tip} = Y_{codo} + L_2 \cdot \sin(\theta_1 + \theta_2)\]

---

## 2. Arquitectura de Archivos a Generar

Escribe de forma completa (sin secciones omitidas ni placeholders del tipo `...` o `pass`) los siguientes 3 archivos:

```text
Aprendizaje_Por_Refuerzo/05_escorpion/
│
├── scorpion_env.py      # Entorno Gymnasium (Física cinemática continua del escorpión).
├── train_scorpion.py    # Bucle de entrenamiento usando PPO de SB3 optimizado en CUDA.
└── main.py              # Visualizador interactivo ASCII del brazo golpeando el objetivo.
```

### A. `scorpion_env.py` (Entorno de Control Continuo)
*   **Clase Base:** Hereda directamente de `gymnasium.Env`.
*   **Espacio de Estados (`observation_space`):** Un `Box` continuo de 6 elementos: 
    `[sin(theta1), cos(theta1), sin(theta2), cos(theta2), target_x, target_y]`.
    El objetivo flotante aparece en una posición aleatoria del semicírculo superior en cada `reset()`.
*   **Espacio de Acciones (`action_space`):** Un `Box` continuo de 2 elementos acotados de forma estricta entre `[-1.0, 1.0]`. Representan los *torques* (fuerzas angulares de rotación) aplicados directamente a `theta1` y `theta2`.
*   **Dinámica del Turno (`step`):**
    *   Cada torque modifica las velocidades angulares instantáneas ($\omega_1, \omega_2$) aplicando una fricción pasiva para amortiguar el sistema.
    *   Actualiza los ángulos integrando el paso de tiempo (`dt = 0.05`). Limita las articulaciones para evitar que colapsen sobre sí mismas si es necesario.
    *   Máximo 150 pasos por episodio.
*   **Función de Recompensa (Reward Shaping):**
    *   Calcula la distancia euclidiana entre el aguijón \((X_{tip}, Y_{tip})\) y el objetivo.
    *   **Recompensa densa por cercanía:** Otorga `-distancia`. Entre más cerca esté, más se aproxima la recompensa a 0.
    *   **Premio por Impacto Extremo:** Si la distancia es menor a un radio de tolerancia de 1.5 unidades, otorga un bono plano de `+10.0` y marca `terminated = True`.
    *   **Costo de Energía:** Resta `-0.01 * (torque1^2 + torque2^2)` para desincentivar movimientos espasmódicos y forzar trayectorias orgánicas y fluidas.

### B. `train_scorpion.py` (Entrenamiento con PPO)
*   Instancia el entorno `ScorpionEnv`.
*   Configura el agente `PPO` de Stable-Baselines3 utilizando la política continua `MlpPolicy`.
*   **Hiperparámetros óptimos para control motor:** `learning_rate=0.0003`, `n_steps=2048`, `batch_size=64`, `gae_lambda=0.95`, `gamma=0.99`.
*   **Hardware:** Fuerza de forma obligatoria el dispositivo `device="cuda"` validando previamente con PyTorch (`torch.cuda.is_available()`).
*   Entrena al agente durante **80,000 pasos de tiempo** globales y guarda el archivo binario en `models/scorpion_ppo_model.zip`.

### C. `main.py` (Motor de Proyección Gráfica ASCII)
*   Carga la política desde `models/scorpion_ppo_model.zip`.
*   Ejecuta simulaciones en bucle interactivo infinito en modo puramente determinista.
*   **Proyección de Matriz de Texto:** Limpia la pantalla en cada frame (`time.sleep(0.08)`). Mapea el espacio continuo en una cuadrícula discreta de texto (ej. 50 × 25 caracteres). Dibuja la trayectoria uniendo los puntos (0,0), \((X_{codo}, Y_{codo})\) y \((X_{tip}, Y_{tip})\) usando caracteres de enlace (como `#` o `o`), representa el objetivo flotante con una `🎯` o una `X`, y muestra los torques instantáneos aplicados por el modelo en la parte inferior.

---

## 3. Requisitos de Código y Estilo

1.  **Sintaxis Moderna de Gymnasium:** Cumple estrictamente el protocolo devolviendo 5 elementos en `step` (`obs, reward, terminated, truncated, info`) y 2 en `reset` (`obs, info`).
2.  **Cero Placeholders:** Genera los 3 archivos de código fuente de inicio a fin con todas sus importaciones matemáticas de NumPy, PyTorch y trigonometría básica listas para su uso.
