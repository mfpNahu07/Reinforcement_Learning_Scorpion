import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ScorpionEnv(gym.Env):
    """
    Entorno Gymnasium para El Escorpión Luchador (Péndulo doble).
    Física cinemática continua con 2 grados de libertad.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(ScorpionEnv, self).__init__()
        
        # Parámetros del brazo articulado
        self.L1 = 15.0
        self.L2 = 10.0
        self.dt = 0.05
        self.max_steps = 300
        self.current_step = 0
        
        # Estado interno
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.omega1 = 0.0
        self.omega2 = 0.0
        self.target_x = 0.0
        self.target_y = 0.0

        # Espacio de Acciones: torques limitados entre [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Espacio de Estados: [sin(t1), cos(t1), sin(t2), cos(t2), omega1, omega2, target_x, target_y]
        high = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 25.0, 25.0], dtype=np.float32)
        low = np.array([-1.0, -1.0, -1.0, -1.0, -5.0, -5.0, -25.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_obs(self):
        return np.array([
            np.sin(self.theta1),
            np.cos(self.theta1),
            np.sin(self.theta2),
            np.cos(self.theta2),
            self.omega1,
            self.omega2,
            self.target_x,
            self.target_y
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Inicializar el escorpión apuntando hacia arriba (aprox)
        self.theta1 = self.np_random.uniform(np.pi/4, 3*np.pi/4)
        self.theta2 = self.np_random.uniform(-np.pi/4, np.pi/4)
        self.omega1 = 0.0
        self.omega2 = 0.0
        
        # Generar objetivo aleatorio en semicírculo superior
        # radio entre 10 y 24, ángulo entre 0 y pi
        radio = self.np_random.uniform(10.0, 24.0)
        angulo = self.np_random.uniform(0.0, np.pi)
        self.target_x = radio * np.cos(angulo)
        self.target_y = radio * np.sin(angulo)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        torque1, torque2 = action
        
        # Dinámica del movimiento: actualización de velocidades con fricción pasiva
        friccion = 0.1
        self.omega1 = self.omega1 * (1.0 - friccion) + torque1 * self.dt
        self.omega2 = self.omega2 * (1.0 - friccion) + torque2 * self.dt
        
        # Limitar la velocidad angular máxima para evitar descontrol
        self.omega1 = np.clip(self.omega1, -5.0, 5.0)
        self.omega2 = np.clip(self.omega2, -5.0, 5.0)
        
        # Actualización de posiciones
        self.theta1 += self.omega1 * self.dt
        self.theta2 += self.omega2 * self.dt
        
        # Cinemática Directa
        x_codo = self.L1 * np.cos(self.theta1)
        y_codo = self.L1 * np.sin(self.theta1)
        x_tip = x_codo + self.L2 * np.cos(self.theta1 + self.theta2)
        y_tip = y_codo + self.L2 * np.sin(self.theta1 + self.theta2)
        
        # Calcular distancia al objetivo
        distancia = np.sqrt((x_tip - self.target_x)**2 + (y_tip - self.target_y)**2)
        
        # Función de Recompensa (Reward Shaping)
        # 1. Recompensa densa por cercanía
        reward = -distancia
        
        # 2. Costo de energía para movimientos fluidos
        costo_energia = 0.01 * (torque1**2 + torque2**2)
        reward -= costo_energia
        
        # 3. Castigo por alta velocidad al acercarse al objetivo (para que aprenda a frenar suavemente)
        if distancia < 4.0:
            reward -= 0.5 * (self.omega1**2 + self.omega2**2)
        
        # 4. Premio por impacto extremo
        terminated = False
        if distancia < 1.5:
            reward += 10.0
            terminated = True
            
        # Condición de fin por pasos
        truncated = self.current_step >= self.max_steps
        
        info = {
            "x_codo": x_codo,
            "y_codo": y_codo,
            "x_tip": x_tip,
            "y_tip": y_tip,
            "distance": distancia
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info
