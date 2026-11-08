"""
Configuração central do projeto ALPR.
Ajuste estes parâmetros conforme a máquina onde o script for rodar.
"""

import os

# ---------- Caminhos ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "data", "videos_entrada")       # vídeos baixados da câmera, ainda não processados
VIDEOS_PROCESSADOS_DIR = os.path.join(BASE_DIR, "data", "videos_processados")  # vídeos já processados (movidos pra cá)
FOTOS_DIR = os.path.join(BASE_DIR, "data", "fotos")                 # fotos dos veículos detectados
PLANILHA_PATH = os.path.join(BASE_DIR, "data", "placas_detectadas.xlsx")
PROCESSED_LOG = os.path.join(BASE_DIR, "data", "videos_processados.log")  # controle de vídeos já processados

# ---------- Modo de Operação ----------
# "total_scan" -> MODO VARREDURA TOTAL: rastreia e registra TODOS os veículos da cena
#                 (inclusive parados, estacionados, sem placa visível ou placa ilegível)
# "plate_only" -> Modo clássico: processa e registra apenas veículos com placa detectada
OPERATION_MODE = "total_scan"

# Parâmetros de filtro para Varredura Total
MIN_VEHICLE_WIDTH = 80       # largura mínima em pixels no frame reduzido (descarta ruídos/carros no horizonte)
MIN_VEHICLE_HEIGHT = 80      # altura mínima em pixels no frame reduzido
MIN_VEHICLE_FRAMES = 3       # veículo deve ser detectado por ao menos N frames para evitar falsos positivos momentâneos

# ---------- Modelo de detecção ----------
# MODEL_TYPE define como o Detector interpreta as detecções:
#   "plate"   -> modelo treinado especificamente pra detectar PLACAS (1 classe).
#                 Todas as detecções já são placas, sem precisar filtrar.
#   "vehicle" -> modelo genérico (ex: yolov8n.pt do COCO), detecta o VEÍCULO;
#                 o OCR roda em cima do crop do veículo inteiro (menos preciso).
MODEL_TYPE = "plate"

# Caminho do modelo. Depois de treinar (veja train_plate_model.py), o arquivo
# gerado fica em runs/detect/placa_detector/weights/best.pt — copie ele pra
# models/placa_yolov8n.pt (ou ajuste o caminho abaixo).
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "placa_yolov8n.pt")

# Usado apenas quando MODEL_TYPE = "vehicle" (modelo genérico COCO).
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Confiança mínima pra considerar uma detecção válida
YOLO_CONFIDENCE = 0.4

# ---------- Processamento de vídeo ----------
FRAME_SKIP = 2          # processa 1 a cada 2 frames, reduzindo o tempo pela metade mantendo RESIZE_WIDTH = 1280 para não perder resolução da placa
RESIZE_WIDTH = 1280      # redimensiona o frame antes de processar (None = mantém resolução original)
USE_GPU = True           # tenta usar GPU (CUDA) se disponível; cai pra CPU automaticamente se não tiver

# ---------- OCR ----------
OCR_ENGINE = "fast_plate_ocr"   # "fast_plate_ocr", "easyocr" ou "paddleocr"
OCR_LANGUAGES = ["en"]   # EasyOCR usa "en" mesmo pra placas BR (são caracteres latinos/números)
OCR_MIN_CONFIDENCE = 0.1

# ---------- Validação de placa (Brasil) ----------
# Formato antigo: ABC1234 | Formato Mercosul: ABC1D23
REGEX_PLACA_ANTIGA = r"^[A-Z]{3}[0-9]{4}$"
REGEX_PLACA_MERCOSUL = r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$"

# ---------- Deduplicação ----------
# Não grava a mesma placa de novo se detectada dentro dessa janela de tempo (segundos)
DEDUP_WINDOW_SECONDS = 60

# ---------- Classificação de Marca e Modelo (Fase 2 - VMMR) ----------
CAR_CLASSIFIER_ENABLED = True
# Motor de classificação:
#   "siglip" -> Zero-Shot com Google SigLIP + Prompt Ensembling (100% offline após 1º download, sem precisar treinar)
#   "yolo"   -> Classificador supervisionado YOLOv8-cls (models/car_yolov8_cls.pt)
CAR_CLASSIFIER_ENGINE = "siglip"

# Modelo SigLIP (opções recomendadas):
#   "google/siglip-base-patch16-224"     (~800 MB - rápido, roda em CPU e GPU)
#   "google/siglip-base-patch16-384"     (~800 MB - maior resolução 384x384, excelente distinção de detalhes)
#   "google/siglip-so400m-patch14-384"   (~3.5 GB - altíssima precisão em 384x384, ideal para GPU NVIDIA)
SIGLIP_MODEL_ID = "google/siglip-base-patch16-384"
SIGLIP_CACHE_PATH = os.path.join(BASE_DIR, "models", "siglip_text_embeddings.pt")

# ---------- Matriz de Confiabilidade VMMR ----------
# Parâmetros para determinação da confiança da classificação (Alta, Média, Baixa / Ambígua)
VMMR_CONF_ALTA_MIN = 0.40       # Score do 1º colocado para Confiabilidade Alta (mínimo 40%)
VMMR_MARGEM_ALTA = 0.25         # Diferença mínima entre 1º e 2º para Confiabilidade Alta (prob_1 - prob_2 >= 0.25)
VMMR_MARGEM_MEDIA = 0.10        # Diferença mínima entre 1º e 2º para Confiabilidade Média (prob_1 - prob_2 >= 0.10)

# Caminho opcional caso use CAR_CLASSIFIER_ENGINE = "yolo"
CAR_MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "car_yolov8_cls.pt")
CAR_CLASSIFIER_MIN_CONF = 0.08
