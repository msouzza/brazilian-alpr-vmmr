"""
Treina um modelo YOLOv8 de Classificação (YOLOv8-cls) para Reconhecimento
de Marca e Modelo de Veículos (VMMR) focado no mercado brasileiro (modelos nacionais).

Exemplos de classes suportadas:
    - Volkswagen Fox, Gol, Polo, Voyage, T-Cross
    - Chevrolet Onix, Prisma, Celta, Tracker
    - Fiat Palio, Uno, Strada, Toro, Mobi, Argo
    - Hyundai HB20, Creta
    - Ford Ka, Fiesta, EcoSport
    - Toyota Corolla, Hilux, Etios
    - Honda Civic, Fit, HR-V
    - Renault Sandero, Logan, Duster
    - Jeep Renegade, Compass

Uso:
    $env:ROBOFLOW_API_KEY="sua_chave_aqui"
    python train_vmmr_model.py

ATENÇÃO:
    Este treinamento é executado APENAS UMA VEZ!
    O resultado (best.pt) é copiado para 'models/car_yolov8_cls.pt' e exportado
    para 'models/car_model.onnx'. O pipeline principal (main.py) carrega esses
    pesos automaticamente de forma definitiva para todas as execuções futuras.
"""

import os
import shutil
import cv2
import yaml
from ultralytics import YOLO

# ---------- Configurações de Treinamento ----------
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "BSqnEpUnz6BLi5Lcswme")

RAW_DATASET_DIR = "dataset_raw_yolo"
DATASET_DIR = "dataset_vmmr"
EPOCHS = 40
IMG_SIZE = 224             # Resolução padrão ótima para redes de classificação de veículos
MODELO_BASE = "yolov8n-cls.pt"  # Nano: ultra leve (~5MB), rápido e roda perfeitamente em GPUs comuns e CPUs

DESTINO_MODELO_PT = os.path.join("models", "car_yolov8_cls.pt")
DESTINO_MODELO_ONNX = os.path.join("models", "car_model.onnx")


def converter_deteccao_para_classificacao(pasta_raw: str, pasta_saida: str):
    """
    Lê as anotações do dataset de detecção YOLOv8 (data.yaml, images, labels)
    e extrai os crops dos veículos organizados em pastas por classe para treino do YOLOv8-cls:
        pasta_saida/
          train/
            vw-fox/
            ...
          val/
            vw-fox/
            ...
    """
    print("[Treino VMMR] Convertendo detecções em recortes de veículos para classificação...")
    yaml_path = os.path.join(pasta_raw, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Arquivo data.yaml não encontrado em {pasta_raw}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_info = yaml.safe_load(f)

    classes = data_info.get("names", [])
    if isinstance(classes, dict):
        classes = [classes[k] for k in sorted(classes.keys())]

    splits = [("train", "train"), ("valid", "val"), ("test", "val")]

    total_crops = 0
    for split_origem, split_destino in splits:
        img_dir = os.path.join(pasta_raw, split_origem, "images")
        lbl_dir = os.path.join(pasta_raw, split_origem, "labels")

        if not os.path.exists(img_dir):
            continue

        for img_nome in os.listdir(img_dir):
            if not img_nome.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            base_nome = os.path.splitext(img_nome)[0]
            lbl_path = os.path.join(lbl_dir, base_nome + ".txt")
            if not os.path.exists(lbl_path):
                continue

            img_path = os.path.join(img_dir, img_nome)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            with open(lbl_path, "r", encoding="utf-8") as lf:
                linhas = lf.readlines()

            for idx, linha in enumerate(linhas):
                partes = linha.strip().split()
                if len(partes) < 5:
                    continue

                cls_id = int(partes[0])
                if cls_id >= len(classes):
                    continue

                cls_nome = classes[cls_id]
                x_center, y_center, bw, bh = map(float, partes[1:5])

                # Converte coordenadas normalizadas para pixels
                x1 = max(0, int((x_center - bw / 2) * w))
                y1 = max(0, int((y_center - bh / 2) * h))
                x2 = min(w, int((x_center + bw / 2) * w))
                y2 = min(h, int((y_center + bh / 2) * h))

                if (x2 - x1) < 25 or (y2 - y1) < 25:
                    continue

                crop = img[y1:y2, x1:x2]

                pasta_classe = os.path.join(pasta_saida, split_destino, cls_nome)
                os.makedirs(pasta_classe, exist_ok=True)

                caminho_crop = os.path.join(pasta_classe, f"{base_nome}_c{idx}.jpg")
                cv2.imwrite(caminho_crop, crop)
                total_crops += 1

    print(f"[Treino VMMR] Conversão concluída! {total_crops} recortes gerados em '{pasta_saida}/'.")


def verificar_e_preparar_brcars() -> bool:
    """
    Verifica se o usuário descompactou o dataset BRCars em 'brcars196', 'dataset_brcars' ou 'brcars427'.
    Lê o arquivo CSV de metadados (brcars196.csv ou brcars427.csv) e organiza as fotos externas (persp == 'e')
    na pasta 'dataset_vmmr/train/' e 'dataset_vmmr/val/'.
    """
    candidatos_pastas = ["brcars196", "brcars427", "dataset_brcars", "brcars"]
    pasta_brcars = None
    for cand in candidatos_pastas:
        if os.path.exists(cand):
            pasta_brcars = cand
            break

    if pasta_brcars is None:
        return False

    candidatos_csv = [
        os.path.join(pasta_brcars, "brcars196.csv"),
        os.path.join(pasta_brcars, "brcars427.csv"),
        "brcars196.csv",
        "brcars427.csv",
    ]
    csv_path = None
    for c in candidatos_csv:
        if os.path.exists(c):
            csv_path = c
            break

    if csv_path is None:
        print(f"[Treino VMMR] Pasta '{pasta_brcars}' encontrada, mas o CSV (brcars196.csv) não foi localizado.")
        return False

    try:
        import pandas as pd
    except ImportError:
        print("[Treino VMMR] Instalando pandas para leitura do CSV do BRCars...")
        os.system("pip install pandas")
        import pandas as pd

    print(f"\n[Treino VMMR] Base BRCars detectada em '{pasta_brcars}' com metadados em '{csv_path}'!")
    print("[Treino VMMR] Filtrando imagens de perspectiva externa (persp == 'e') para montar o dataset...")

    df = pd.read_csv(csv_path)
    if "persp" in df.columns:
        df = df[df["persp"] == "e"]

    total = 0
    for _, row in df.iterrows():
        make = str(row.get("make_label", "")).strip().lower().replace(" ", "_")
        model = str(row.get("model_label", "")).strip().lower().replace(" ", "_")
        classe = f"{make}_{model}"
        split_num = int(row.get("split", 0))
        split_dir = "val" if split_num in (1, 2) else "train"

        uri = str(row.get("uri", "")).replace("\\", "/")
        caminho_origem = os.path.join(pasta_brcars, uri)
        if not os.path.exists(caminho_origem):
            caminho_origem = os.path.join(pasta_brcars, os.path.basename(uri))

        if not os.path.exists(caminho_origem):
            continue

        pasta_destino = os.path.join(DATASET_DIR, split_dir, classe)
        os.makedirs(pasta_destino, exist_ok=True)
        nome_arquivo = os.path.basename(uri)
        caminho_destino = os.path.join(pasta_destino, nome_arquivo)

        if not os.path.exists(caminho_destino):
            shutil.copy(caminho_origem, caminho_destino)
            total += 1

    print(f"[Treino VMMR] BRCars estruturado com sucesso! {total} imagens externas organizadas em '{DATASET_DIR}/'.")
    return total > 0


def preparar_dataset():
    """
    1. Verifica se o BRCars foi descompactado localmente (brcars196/ ou dataset_brcars/).
    2. Se já existe 'dataset_vmmr/train', reutiliza diretamente.
    3. Fallback: Baixa do Roboflow Universe no formato 'yolov8'.
    """
    # 1. Prioridade máxima: BRCars descompactado pelo usuário
    if verificar_e_preparar_brcars():
        return DATASET_DIR

    # 2. Dataset já preparado anteriormente
    if os.path.exists(DATASET_DIR) and os.path.exists(os.path.join(DATASET_DIR, "train")):
        print(f"[Treino VMMR] Dataset de classificação encontrado em '{DATASET_DIR}/'.")
        return DATASET_DIR

    if ROBOFLOW_API_KEY == "COLE_SUA_API_KEY_AQUI" or not ROBOFLOW_API_KEY.strip():
        raise RuntimeError(
            "Defina sua API key do Roboflow (variável ROBOFLOW_API_KEY no terminal ou no script)."
        )

    from roboflow import Roboflow
    print("[Treino VMMR] Conectando ao Roboflow com a chave fornecida...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    projeto = rf.workspace("vehiclemodeldetection").project("vehicle-model-detection-2")
    versao = projeto.version(1)

    print("[Treino VMMR] Baixando dataset no formato 'yolov8'...")
    dataset = versao.download("yolov8", location=RAW_DATASET_DIR)

    # Converte detecções em recortes de classificação
    converter_deteccao_para_classificacao(dataset.location, DATASET_DIR)

    return DATASET_DIR


def treinar(dataset_path: str):
    print(f"\n[Treino VMMR] Inicializando treinamento YOLOv8-cls ({MODELO_BASE})...")
    print(f"  - Épocas: {EPOCHS}")
    print(f"  - Img Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Dataset: {dataset_path}")

    modelo = YOLO(MODELO_BASE)

    resultado = modelo.train(
        data=dataset_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        name="car_vmmr_classifier",
        patience=10,
        batch=32,
    )

    return resultado


def salvar_pesos_finais():
    origem_pt = os.path.join("runs", "classify", "car_vmmr_classifier", "weights", "best.pt")

    if not os.path.exists(origem_pt):
        print(f"[Treino VMMR] Aviso: '{origem_pt}' não encontrado diretamente.")
        pasta_runs = os.path.join("runs", "classify")
        if os.path.exists(pasta_runs):
            subpastas = sorted(os.listdir(pasta_runs), reverse=True)
            for sub in subpastas:
                candidato = os.path.join(pasta_runs, sub, "weights", "best.pt")
                if os.path.exists(candidato):
                    origem_pt = candidato
                    break

    if os.path.exists(origem_pt):
        os.makedirs("models", exist_ok=True)
        shutil.copy(origem_pt, DESTINO_MODELO_PT)
        print(f"[Treino VMMR] Modelo PyTorch copiado com sucesso para: {DESTINO_MODELO_PT}")

        # Exporta opcionalmente para ONNX (inferência acelerada)
        try:
            print("[Treino VMMR] Exportando modelo para formato ONNX...")
            modelo_treinado = YOLO(DESTINO_MODELO_PT)
            modelo_treinado.export(format="onnx", imgsz=IMG_SIZE)
            onnx_gerado = DESTINO_MODELO_PT.replace(".pt", ".onnx")
            if os.path.exists(onnx_gerado) and onnx_gerado != DESTINO_MODELO_ONNX:
                shutil.move(onnx_gerado, DESTINO_MODELO_ONNX)
                print(f"[Treino VMMR] Modelo ONNX salvo em: {DESTINO_MODELO_ONNX}")
        except Exception as e:
            print(f"[Treino VMMR] Aviso na exportação ONNX (o arquivo .pt continuará utilizável): {e}")

        print("\n[Treino VMMR] Concluído com sucesso! O main.py já utilizará os novos pesos automaticamente.")
    else:
        print("[Treino VMMR] Não foi possível localizar o arquivo 'best.pt' gerado pelo Ultralytics.")


if __name__ == "__main__":
    caminho_dados = preparar_dataset()
    treinar(caminho_dados)
    salvar_pesos_finais()
