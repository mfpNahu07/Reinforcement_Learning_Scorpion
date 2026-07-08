# 🦂 Escorpión Luchador: Simulación Robótica con IA

Este proyecto es un simulador de brazo robótico ("Escorpión Luchador") controlado por una Inteligencia Artificial entrenada mediante **Aprendizaje por Refuerzo (Reinforcement Learning)**. 

La IA utiliza el algoritmo **PPO (Proximal Policy Optimization)** para aprender a controlar cinemáticamente las articulaciones del brazo de forma continua y en tiempo real, con el objetivo de hacer que el "aguijón" alcance con precisión un objetivo dinámico en el espacio.

Además, incluye un dashboard interactivo de telemetría web construido con **FastAPI**, **WebSockets** y **HTML5 Canvas** (estilizado con Tailwind CSS) para visualizar la simulación, los torques aplicados y la evolución de los ángulos en vivo.

## 🚀 Tecnologías Utilizadas

* **Python 3**
* **Stable-Baselines3** (Algoritmo PPO)
* **Gymnasium** (Entorno de simulación de aprendizaje por refuerzo)
* **PyTorch** (Motor de redes neuronales)
* **FastAPI & Uvicorn** (Servidor web asíncrono)
* **WebSockets** (Transmisión de telemetría en tiempo real)
* **HTML5 Canvas & Tailwind CSS** (Interfaz visual)

---

## 🛠️ Instalación y Requisitos

1. Asegúrate de tener Python instalado en tu sistema.
2. Clona este repositorio y navega hasta esta carpeta.
3. Se recomienda crear un entorno virtual e instalar las dependencias desde el archivo `requirements.txt`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## ⚙️ Cómo Entrenar y Ejecutar el Proyecto

Para correr este proyecto, sigue estos tres simples pasos desde tu terminal (en Windows):

### 1. Activar el Entorno Virtual
Antes de ejecutar cualquier script, asegúrate de activar el entorno virtual donde están instaladas las dependencias:
```powershell
.venv\Scripts\Activate
```

### 2. Entrenar el Modelo de IA
Para que el escorpión aprenda a moverse, debes ejecutar el script de entrenamiento. Esto generará y guardará el modelo entrenado (`scorpion_ppo_model.zip`) en la carpeta `models/`.
```powershell
python train_scorpion.py
```
*(Nota: El entrenamiento puede tardar unos minutos dependiendo de la potencia de tu CPU/GPU).*

### 3. Lanzar el Servidor y la Interfaz Visual
Una vez finalizado el entrenamiento, puedes iniciar el servidor web para ver al agente en acción. 
```powershell
python server_ui.py
```
Al ejecutar este comando, el servidor de telemetría arrancará. Abre tu navegador web y dirígete a [http://127.0.0.1:8001/](http://127.0.0.1:8001/) para ver el Dashboard Robótico.

---

## 💡 Solución de Problemas (Troubleshooting)

Si por alguna razón la aplicación se queda trabada o te marca que el puerto está en uso, puedes forzar el cierre del proceso ocupando el puerto en Windows abriendo PowerShell como Administrador y ejecutando:

```powershell
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8001).OwningProcess -Force
```
*(Asegúrate de cambiar el `8001` por `8000` u otro si cambiaste el puerto predeterminado del servidor).*
