import os
import torch
from stable_baselines3 import PPO
from scorpion_env import ScorpionEnv

def main():
    # Validar hardware
    is_cuda_available = torch.cuda.is_available()
    device = "cuda" if is_cuda_available else "cpu"
    print(f"CUDA Disponible: {is_cuda_available}")
    
    # El prompt requiere forzar device="cuda"
    device_to_use = "cuda"
    print(f"Forzando el uso de dispositivo: {device_to_use}")

    # Crear la carpeta models si no existe
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Instanciar el entorno
    env = ScorpionEnv()

    # Configurar el modelo PPO de SB3
    # Hiperparámetros óptimos requeridos para control motor
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        device=device_to_use,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    print("Iniciando entrenamiento por 500,000 pasos de tiempo...")
    model.learn(total_timesteps=500000)

    # Guardar el modelo
    save_path = os.path.join(models_dir, "scorpion_ppo_model")
    model.save(save_path)
    print(f"Entrenamiento completado. Modelo guardado en {save_path}.zip")

if __name__ == "__main__":
    main()
