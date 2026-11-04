"""
Treina um modelo YOLOv8 especificamente pra detecção de placas veiculares,
usando o dataset público "License Plate Recognition" do Roboflow Universe
(licença CC BY 4.0): https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e

RODAR ESTE SCRIPT UMA VEZ SÓ, LOCALMENTE (não precisa rodar de novo depois).
O resultado (best.pt) é copiado pra models/placa_yolov8n.pt e usado por main.py.

---------------------------------------------------------------------------
PASSO 1 - Criar conta gratuita no Roboflow e pegar sua API key:
    https://app.roboflow.com/  -> Settings -> API Keys

PASSO 2 - Instalar a lib do Roboflow (só precisa pra baixar o dataset):
    pip install roboflow

PASSO 3 - Colar sua API key abaixo (ou setar como variável de ambiente
    ROBOFLOW_API_KEY) e rodar:
    python train_plate_model.py
---------------------------------------------------------------------------
"""

import os
import shutil
from ultralytics import YOLO

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "COLE_SUA_API_KEY_AQUI")

DATASET_DIR = "dataset_placas"
EPOCHS = 60
IMG_SIZE = 640
MODELO_BASE = "yolov8n.pt"  # nano: mais rápido, ideal pra 1 classe só (placa)

MODELO_FINAL_DESTINO = os.path.join("models", "placa_yolov8n.pt")


def baixar_dataset():
    if os.path.exists(DATASET_DIR):
        print(f"[Treino] Dataset já existe em '{DATASET_DIR}/', pulando download.")
        return DATASET_DIR

    if ROBOFLOW_API_KEY == "COLE_SUA_API_KEY_AQUI":
        raise RuntimeError(
            "Defina sua API key do Roboflow (variável ROBOFLOW_API_KEY ou "
            "edite a constante no topo deste arquivo). Crie uma conta gratuita "
            "em https://app.roboflow.com/ para obter a sua."
        )

    from roboflow import Roboflow
    print("[Treino] Baixando dataset do Roboflow Universe...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    projeto = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    versao = projeto.version(4)
    dataset = versao.download("yolov8", location=DATASET_DIR)
    return dataset.location


def treinar(data_yaml_path: str):
    print(f"[Treino] Iniciando treino YOLOv8 ({MODELO_BASE}) por {EPOCHS} épocas...")
    modelo = YOLO(MODELO_BASE)

    resultado = modelo.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        name="placa_detector",
        patience=15,   # early stopping se não melhorar por 15 épocas
    )

    return resultado


def copiar_modelo_final():
    origem = os.path.join("runs", "detect", "placa_detector", "weights", "best.pt")

    if not os.path.exists(origem):
        # Se já rodou antes, o Ultralytics cria placa_detector2, placa_detector3, etc.
        print(f"[Treino] AVISO: não achei '{origem}'. Procure manualmente a pasta "
              f"mais recente em runs/detect/ e copie o best.pt pra {MODELO_FINAL_DESTINO}")
        return

    os.makedirs("models", exist_ok=True)
    shutil.copy(origem, MODELO_FINAL_DESTINO)
    print(f"[Treino] Modelo final copiado pra: {MODELO_FINAL_DESTINO}")
    print("[Treino] Pronto! O config.py já aponta pra esse caminho por padrão.")


if __name__ == "__main__":
    dataset_path = baixar_dataset()
    data_yaml = os.path.join(dataset_path, "data.yaml")
    treinar(data_yaml)
    copiar_modelo_final()
