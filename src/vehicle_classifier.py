"""
Módulo de classificação de veículos (Fase 2):
- Estimativa da cor da lataria (espaço HSV com correção de iluminação)
- Mapeamento de tipo de veículo (Carro, Moto, Caminhão, Ônibus)
- Classificação de Marca, Modelo e Categoria de Carroceria (Hatch, Sedan, SUV, Minivan, Picape)
"""

import os
import sys
import cv2
import numpy as np
import torch
import torchvision
from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Mapeamento de classes do modelo COCO para português
MAPA_TIPOS_VEICULO = {
    2: "Carro",
    3: "Moto",
    5: "Ônibus",
    7: "Caminhão",
}

# Mapeamento de modelos brasileiros para categorias de carroceria
MAPA_MODELOS_CATEGORIA = {
    # Hatchbacks populares BR
    "fox": "Hatchback",
    "gol": "Hatchback",
    "palio": "Hatchback",
    "uno": "Hatchback",
    "mobi": "Hatchback",
    "up": "Hatchback",
    "celta": "Hatchback",
    "onix": "Hatchback",
    "hb20": "Hatchback",
    "ka": "Hatchback",
    "polo": "Hatchback",
    "fit": "Hatchback",
    "sandero": "Hatchback",
    "clio": "Hatchback",
    "208": "Hatchback",
    "206": "Hatchback",
    "207": "Hatchback",
    "c3": "Hatchback",
    "fiesta": "Hatchback",
    "etios": "Hatchback",
    "yaris": "Hatchback",
    "golf": "Hatchback",
    "cruze sport6": "Hatchback",
    "focus": "Hatchback",

    # Sedans
    "voyage": "Sedan",
    "siena": "Sedan",
    "grand siena": "Sedan",
    "cronos": "Sedan",
    "prisma": "Sedan",
    "joy plus": "Sedan",
    "onix plus": "Sedan",
    "hb20s": "Sedan",
    "corolla": "Sedan",
    "civic": "Sedan",
    "cruze": "Sedan",
    "jetta": "Sedan",
    "passat": "Sedan",
    "sentra": "Sedan",
    "versa": "Sedan",
    "city": "Sedan",
    "virtus": "Sedan",
    "logan": "Sedan",
    "fiesta sedan": "Sedan",
    "classic": "Sedan",
    "cobalt": "Sedan",
    "etios sedan": "Sedan",

    # SUVs / Utilitários
    "ecosport": "SUV / Utilitário",
    "duster": "SUV / Utilitário",
    "kicks": "SUV / Utilitário",
    "creta": "SUV / Utilitário",
    "t-cross": "SUV / Utilitário",
    "nivus": "SUV / Utilitário",
    "taos": "SUV / Utilitário",
    "renegade": "SUV / Utilitário",
    "compass": "SUV / Utilitário",
    "commander": "SUV / Utilitário",
    "tracker": "SUV / Utilitário",
    "hr-v": "SUV / Utilitário",
    "wr-v": "SUV / Utilitário",
    "cr-v": "SUV / Utilitário",
    "tiguan": "SUV / Utilitário",
    "corolla cross": "SUV / Utilitário",
    "pulse": "SUV / Utilitário",
    "fastback": "SUV / Utilitário",
    "tucson": "SUV / Utilitário",
    "ix35": "SUV / Utilitário",
    "santa fe": "SUV / Utilitário",
    "sw4": "SUV / Utilitário",
    "pajero": "SUV / Utilitário",

    # Picapes
    "strada": "Picape",
    "saveiro": "Picape",
    "toro": "Picape",
    "hilux": "Picape",
    "s10": "Picape",
    "ranger": "Picape",
    "amarok": "Picape",
    "l200": "Picape",
    "oroch": "Picape",
    "montana": "Picape",
    "courier": "Picape",
    "ram": "Picape",

    # Minivans
    "spin": "Minivan",
    "zafira": "Minivan",
    "meriva": "Minivan",
    "livina": "Minivan",
    "idea": "Minivan",
    "doblo": "Minivan",

    # Vans / Utilitários de carga
    "kombi": "Furgão / Van",
    "fiorino": "Furgão / Van",
    "partner": "Furgão / Van",
    "expert": "Furgão / Van",
    "master": "Furgão / Van",
    "ducato": "Furgão / Van",
    "transit": "Furgão / Van",
    "sprinter": "Furgão / Van",
}


def inferir_categoria_por_modelo(nome_modelo: str) -> str:
    """Inferir a carroceria/categoria a partir do nome ou modelo do veículo."""
    nome_norm = nome_modelo.lower()
    for chave, cat in MAPA_MODELOS_CATEGORIA.items():
        if chave in nome_norm:
            return cat
    return "Carro"


def formatar_nome_veiculo(classe: str) -> str:
    """
    Formata strings como 'vw_fox', 'chevrolet-onix', 'volkswagen_fox_2013'
    para 'Volkswagen Fox', 'Chevrolet Onix', etc.
    """
    limpo = classe.replace("-", " ").replace("_", " ")
    substituicoes = {
        "vw": "Volkswagen",
        "gm": "Chevrolet",
        "chevrolet": "Chevrolet",
        "volkswagen": "Volkswagen",
        "fiat": "Fiat",
        "ford": "Ford",
        "hyundai": "Hyundai",
        "toyota": "Toyota",
        "honda": "Honda",
        "renault": "Renault",
        "nissan": "Nissan",
        "jeep": "Jeep",
        "peugeot": "Peugeot",
        "citroen": "Citroën",
        "citroën": "Citroën",
        "mitsubishi": "Mitsubishi",
    }
    partes = limpo.split()
    partes_fmt = []
    for p in partes:
        p_low = p.lower()
        if p_low in substituicoes:
            partes_fmt.append(substituicoes[p_low])
        elif p.isdigit() and len(p) == 4:
            continue
        else:
            partes_fmt.append(p.capitalize())
    return " ".join(partes_fmt).strip() or classe

# Índices relevantes de veículos no ImageNet 1k
INDICES_VEICULOS_IMAGENET = {
    656: ("Minivan", "Minivan / Monovolume"),
    675: ("Van", "Furgão / Van"),
    436: ("Wagon", "Station Wagon / Perua"),
    468: ("Sedan", "Sedan"),
    609: ("Jeep", "SUV / Utilitário"),
    717: ("Pickup", "Picape"),
    817: ("SportsCar", "Esportivo / Cupê"),
}


def estimar_cor_lataria(crop_veiculo: np.ndarray) -> str:
    """
    Analisa a região representativa da lataria do veículo no espaço de cores HSV
    e retorna a cor predominante em português.
    """
    if crop_veiculo is None or crop_veiculo.size == 0:
        return "Indefinida"

    h, w = crop_veiculo.shape[:2]
    if h < 20 or w < 20:
        return "Indefinida"

    # Seleciona a faixa central da lataria (desconsidera teto/vidro e rodas/asfalto)
    y1, y2 = int(h * 0.30), int(h * 0.75)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    lataria = crop_veiculo[y1:y2, x1:x2]

    if lataria.size == 0:
        lataria = crop_veiculo

    hsv = cv2.cvtColor(lataria, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Filtra reflexos brancos extremos de farol (> 240) e sombras pretas puras (< 30)
    mascara = (V > 30) & (V < 240)
    if np.sum(mascara) < 50:
        mascara = np.ones(V.shape, dtype=bool)

    s_media = float(np.mean(S[mascara]))
    v_media = float(np.mean(V[mascara]))
    h_media = float(np.mean(H[mascara]))

    # Cores acromáticas (baixa saturação: Branco, Preto, Prata / Cinza)
    # Sob iluminação noturna/postes, saturação abaixo de 65 é acromática
    if s_media < 65:
        if v_media < 65:
            return "Preto"
        elif v_media > 145:
            return "Branco"
        else:
            return "Prata / Cinza"

    # Cores cromáticas baseadas no Matiz (Hue no OpenCV vai de 0 a 179)
    if h_media < 12 or h_media >= 165:
        return "Vermelho"
    elif 12 <= h_media < 35:
        return "Amarelo / Laranja"
    elif 35 <= h_media < 85:
        return "Verde"
    elif 85 <= h_media < 135:
        return "Azul"
    elif 135 <= h_media < 165:
        return "Roxo / Vinho"

    return "Outra"


def classificar_tipo_veiculo(cls_id: int) -> str:
    """Retorna o tipo de veículo em português a partir do ID do COCO."""
    return MAPA_TIPOS_VEICULO.get(cls_id, "Carro")


class ClassificadorMarcaModelo:
    """
    Classificador de Marca e Modelo de veículos (Fase 2).
    Suporta:
    1. Modelo YOLOv8 de classificação (ex: models/car_yolov8_cls.pt ou models/car_model.pt)
    2. Modelo customizado BRCars pré-treinado em ONNX (models/car_model.onnx)
    3. Extrator visual MobileNetV3 calibrado para categorias brasileiras (Hatchback, Sedan, SUV, Picape).
    """

    def __init__(self):
        self.habilitado = getattr(config, "CAR_CLASSIFIER_ENABLED", True)
        self.device = "cuda" if (config.USE_GPU and torch.cuda.is_available()) else "cpu"
        self.modelo_yolo_cls = None
        self.modelo_onnx = None
        self.backbone = None

        if not self.habilitado:
            return

        # 1. Tenta carregar modelo YOLOv8-cls especializado (.pt)
        candidatos_pt = [
            os.path.join(config.BASE_DIR, "models", "car_yolov8_cls.pt"),
            os.path.join(config.BASE_DIR, "models", "car_model.pt"),
        ]
        for caminho_pt in candidatos_pt:
            if os.path.exists(caminho_pt):
                try:
                    from ultralytics import YOLO
                    self.modelo_yolo_cls = YOLO(caminho_pt)
                    print(f"[ClassificadorMarcaModelo] Modelo YOLO-cls carregado de: {caminho_pt}")
                    break
                except Exception as e:
                    print(f"[ClassificadorMarcaModelo] Aviso ao carregar YOLO-cls ({caminho_pt}): {e}")

        # 2. Tenta carregar modelo ONNX treinado caso exista
        caminho_weights = getattr(config, "CAR_MODEL_WEIGHTS_PATH", os.path.join(config.BASE_DIR, "models", "car_model.onnx"))
        if self.modelo_yolo_cls is None and os.path.exists(caminho_weights):
            try:
                import onnxruntime as ort
                self.modelo_onnx = ort.InferenceSession(caminho_weights)
                print(f"[ClassificadorMarcaModelo] Pesos ONNX carregados de: {caminho_weights}")
            except Exception as e:
                print(f"[ClassificadorMarcaModelo] Aviso ao carregar ONNX: {e}")

        # 3. Fallback: Carrega MobileNetV3 para extração visual de perfis de carroceria
        try:
            self.backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            ).to(self.device).eval()
        except Exception as e:
            print(f"[ClassificadorMarcaModelo] Aviso ao inicializar MobileNet: {e}")

    def classificar(self, crop_veiculo: np.ndarray, tipo_veiculo: str = "Carro") -> dict:
        """
        Classifica a marca, modelo e categoria do veículo a partir de seu recorte.
        Retorna:
            {"marca_modelo": str, "categoria": str, "confianca": float}
        """
        if not self.habilitado or crop_veiculo is None or crop_veiculo.size == 0:
            return {"marca_modelo": "-", "categoria": tipo_veiculo, "confianca": 0.0}

        h, w = crop_veiculo.shape[:2]
        if h < 40 or w < 40:
            return {"marca_modelo": "-", "categoria": tipo_veiculo, "confianca": 0.0}

        if tipo_veiculo != "Carro":
            return {"marca_modelo": tipo_veiculo, "categoria": tipo_veiculo, "confianca": 1.0}

        # 1. Inferência com YOLOv8-cls (BRCars / Modelos Nacionais)
        if self.modelo_yolo_cls is not None:
            try:
                resultados = self.modelo_yolo_cls(crop_veiculo, verbose=False)
                if resultados and len(resultados) > 0:
                    res = resultados[0]
                    top1_idx = int(res.probs.top1)
                    nome_classe = str(res.names[top1_idx])
                    conf = float(res.probs.top1conf)

                    min_conf = getattr(config, "CAR_CLASSIFIER_MIN_CONF", 0.30)
                    if conf >= min_conf:
                        marca_modelo_fmt = formatar_nome_veiculo(nome_classe)
                        categoria = inferir_categoria_por_modelo(marca_modelo_fmt)
                        return {
                            "marca_modelo": marca_modelo_fmt,
                            "categoria": categoria,
                            "confianca": round(conf, 2),
                        }
            except Exception as e:
                pass

        # 2. Inferência com modelo ONNX customizado
        if self.modelo_onnx is not None:
            try:
                img_rgb = cv2.cvtColor(cv2.resize(crop_veiculo, (224, 224)), cv2.COLOR_BGR2RGB)
                tensor = (img_rgb.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
                nome_entrada = self.modelo_onnx.get_inputs()[0].name
                outputs = self.modelo_onnx.run(None, {nome_entrada: tensor})
                pred_idx = int(np.argmax(outputs[0]))
                conf = float(np.max(outputs[0]))
                return {"marca_modelo": f"Modelo #{pred_idx}", "categoria": "Carro", "confianca": round(conf, 2)}
            except Exception:
                pass

        # 3. Inferência de Carroceria calibrada para mercado brasileiro (MobileNetV3)
        if self.backbone is not None:
            try:
                img_rgb = cv2.cvtColor(cv2.resize(crop_veiculo, (224, 224)), cv2.COLOR_BGR2RGB)
                t = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
                t = torchvision.transforms.functional.normalize(
                    t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.backbone(t)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                p_minivan = float(probs[656])
                p_van = float(probs[675])
                p_pickup = float(probs[717])
                p_jeep = float(probs[609])
                p_sport = float(probs[817])
                p_cab = float(probs[468])

                ratio = w / float(h)

                # Regras de silhueta ajustadas para a frota brasileira:
                # Carros de passeio compactos (Fox, Gol, Palio, Onix) são Hatchbacks.
                if p_pickup > 0.35:
                    categoria = "Picape"
                    conf = p_pickup
                elif p_jeep > 0.35:
                    categoria = "SUV / Utilitário"
                    conf = p_jeep
                elif (p_minivan + p_van) > 0.65 and ratio > 1.50:
                    categoria = "Minivan"
                    conf = max(p_minivan, p_van)
                elif ratio > 1.65 or p_cab > 0.25:
                    categoria = "Sedan"
                    conf = max(p_sport, p_cab, 0.55)
                else:
                    categoria = "Hatchback"
                    conf = 0.70

                return {
                    "marca_modelo": categoria,
                    "categoria": categoria,
                    "confianca": round(conf, 2),
                }
            except Exception:
                pass

        return {"marca_modelo": "-", "categoria": "Carro", "confianca": 0.0}


def classificar_veiculo(crop_veiculo: np.ndarray, cls_id: int = 2,
                        classificador_mm: ClassificadorMarcaModelo | None = None) -> dict:
    """
    Retorna dicionário consolidado com tipo, cor, e marca/modelo do veículo.
    """
    tipo = classificar_tipo_veiculo(cls_id)
    cor = estimar_cor_lataria(crop_veiculo)

    marca_modelo = "-"
    if classificador_mm is not None:
        info_mm = classificador_mm.classificar(crop_veiculo, tipo)
        marca_modelo = info_mm.get("marca_modelo", "-")

    return {
        "tipo": tipo,
        "cor": cor,
        "marca_modelo": marca_modelo,
    }
