"""
Detecção de veículos e placas usando YOLOv8 (Ultralytics).

Duas formas de uso:
1) Modelo genérico (yolov8n.pt) -> detecta o VEÍCULO. A região do veículo é
   recortada e passada pro OCR tentar achar a placa dentro dela (menos preciso,
   mas funciona sem treinar nada).
2) Modelo treinado especificamente em placas -> detecta a PLACA diretamente
   (mais preciso). Basta trocar YOLO_MODEL_PATH no config.py.
"""

import sys
import os
import torch
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Detector:
    def __init__(self):
        self.device = self._resolver_device()
        print(f"[Detector] Carregando modelo de placa '{config.YOLO_MODEL_PATH}' no dispositivo: {self.device}")
        self.model = YOLO(config.YOLO_MODEL_PATH)

        # Modelo para detecção do corpo do veículo (carros, motos, caminhões, ônibus)
        print(f"[Detector] Carregando detector de veículos 'yolov8n.pt' no dispositivo: {self.device}")
        self.model_veiculo = YOLO("yolov8n.pt")

    def _resolver_device(self) -> str:
        if config.USE_GPU and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def detectar_veiculos(self, frame):
        """
        Detecta veículos inteiros no frame (classes COCO: 2=car, 3=motorcycle, 5=bus, 7=truck).
        Retorna lista de dicts: [{"bbox": (x1,y1,x2,y2), "conf": float, "cls": int}, ...]
        """
        resultados = self.model_veiculo.predict(
            frame,
            device=self.device,
            classes=[2, 3, 5, 7],
            conf=0.25,
            verbose=False,
        )

        veiculos = []
        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                veiculos.append({"bbox": (x1, y1, x2, y2), "conf": conf, "cls": cls_id})

        return veiculos

    @staticmethod
    def associar_veiculo_placa(bbox_placa, veiculos, shape_frame):
        """
        Encontra o veículo que engloba a placa. Se nenhum englobar,
        cria uma estimativa geométrica ao redor da placa.
        """
        px1, py1, px2, py2 = bbox_placa
        p_cx = (px1 + px2) / 2
        p_cy = (py1 + py2) / 2

        # 1. Busca veículo que contém o centro da placa
        for v in veiculos:
            vx1, vy1, vx2, vy2 = v["bbox"]
            if vx1 <= p_cx <= vx2 and vy1 <= p_cy <= vy2:
                return v

        # 2. Fallback: expande a região da placa para capturar a lataria ao redor
        h_frame, w_frame = shape_frame[:2]
        pw = px2 - px1
        ph = py2 - py1

        vx1 = max(0, int(px1 - pw * 1.5))
        vy1 = max(0, int(py1 - ph * 3.5))
        vx2 = min(w_frame, int(px2 + pw * 1.5))
        vy2 = min(h_frame, int(py2 + ph * 1.5))

        return {"bbox": (vx1, vy1, vx2, vy2), "conf": 0.5, "cls": 2}

    def detectar(self, frame):
        """
        Roda a detecção em um frame (numpy array BGR, formato OpenCV).
        Retorna lista de dicts: [{"bbox": (x1,y1,x2,y2), "conf": float, "cls": int}, ...]
        """
        resultados = self.model.predict(
            frame,
            device=self.device,
            conf=config.YOLO_CONFIDENCE,
            verbose=False,
        )

        deteccoes = []
        for r in resultados:
            for box in r.boxes:
                cls_id = int(box.cls[0])

                # Modelo genérico (COCO) -> filtra só classes de veículo.
                # Modelo treinado em placas -> não filtra, toda detecção já é placa.
                if config.MODEL_TYPE == "vehicle" and cls_id not in config.VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                deteccoes.append({"bbox": (x1, y1, x2, y2), "conf": conf, "cls": cls_id})

        return deteccoes

    @staticmethod
    def recortar(frame, bbox):
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        return frame[y1:y2, x1:x2]
