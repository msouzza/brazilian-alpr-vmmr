"""
Leitura de texto (OCR) na região recortada da placa.
Suporta EasyOCR (padrão) -- troque OCR_ENGINE no config.py se quiser usar PaddleOCR.
"""

import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LeitorOCR:
    def __init__(self):
        if config.OCR_ENGINE == "easyocr":
            import easyocr
            gpu = config.USE_GPU
            print(f"[OCR] Inicializando EasyOCR (gpu={gpu})")
            self.reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=gpu, verbose=False)
        elif config.OCR_ENGINE == "fast_plate_ocr":
            from fast_plate_ocr import LicensePlateRecognizer
            print("[OCR] Inicializando FastPlateOCR (ONNX Runtime)")
            self.reader = LicensePlateRecognizer("cct-s-v2-global-model")
        elif config.OCR_ENGINE == "paddleocr":
            from paddleocr import PaddleOCR
            print("[OCR] Inicializando PaddleOCR")
            self.reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        else:
            raise ValueError(f"OCR_ENGINE desconhecido: {config.OCR_ENGINE}")

    @staticmethod
    def _preprocessar(imagem):
        """
        Melhora a imagem da placa antes do OCR: amplia se for pequena,
        converte pra tons de cinza e aplica equalização de contraste (CLAHE).
        """
        h, w = imagem.shape[:2]
        if w < 200:
            fator = 200 / w
            imagem = cv2.resize(imagem, (200, int(h * fator)), interpolation=cv2.INTER_CUBIC)

        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(cinza)

    def ler(self, imagem_recortada) -> list[tuple[str, float]]:
        """
        Retorna lista de (texto, confianca) encontrados na imagem recortada.
        """
        if imagem_recortada is None or imagem_recortada.size == 0:
            return []

        resultados = []

        if config.OCR_ENGINE == "fast_plate_ocr":
            # FastPlateOCR espera imagem BGR em 3 canais e já normaliza internamente
            predicoes = self.reader.run(imagem_recortada)
            for pred in predicoes:
                if pred.plate:
                    conf = 0.85 if pred.region == "Brazil" else 0.70
                    resultados.append((pred.plate, conf))
        elif config.OCR_ENGINE == "easyocr":
            imagem_proc = self._preprocessar(imagem_recortada)
            saida = self.reader.readtext(imagem_proc)
            for (_bbox, texto, conf) in saida:
                if conf >= config.OCR_MIN_CONFIDENCE:
                    resultados.append((texto, conf))
        elif config.OCR_ENGINE == "paddleocr":
            imagem_proc = self._preprocessar(imagem_recortada)
            saida = self.reader.ocr(imagem_proc, cls=True)
            if saida and saida[0]:
                for linha in saida[0]:
                    texto, conf = linha[1][0], linha[1][1]
                    if conf >= config.OCR_MIN_CONFIDENCE:
                        resultados.append((texto, conf))
                        resultados.append((texto, conf))

        return resultados
