"""
Ponto de entrada do projeto ALPR.

Uso:
    python main.py

O que faz:
    1. Olha a pasta data/videos_entrada/ em busca de vídeos novos
    2. Pra cada vídeo ainda não processado (controlado por um log local):
        - roda detecção de veículo/placa (YOLOv8)
        - roda OCR na região da placa
        - valida o formato (placa BR)
        - deduplica detecções repetidas
        - salva foto do veículo
        - grava linha na planilha
       - move o vídeo pra pasta de processados

Por enquanto, colocar os vídeos manualmente em data/videos_entrada/
(baixados do cartão SD ou via Wi-Fi da câmera). A automação de download
via API da câmera é o próximo passo, depois que este pipeline estiver
validado.
"""

import os
import shutil

import config
from src.plate_detector import Detector
from src.ocr_reader import LeitorOCR
from src.vehicle_classifier import ClassificadorMarcaModelo
from src.deduplicator import Deduplicador
from src.spreadsheet_writer import PlanilhaWriter
from src.video_processor import processar_video

EXTENSOES_VIDEO = (".mp4", ".mov", ".avi", ".mkv")


def carregar_log_processados() -> set:
    if not os.path.exists(config.PROCESSED_LOG):
        return set()
    with open(config.PROCESSED_LOG, "r", encoding="utf-8") as f:
        return set(linha.strip() for linha in f if linha.strip())


def marcar_como_processado(nome_arquivo: str):
    with open(config.PROCESSED_LOG, "a", encoding="utf-8") as f:
        f.write(nome_arquivo + "\n")


def main():
    os.makedirs(config.VIDEOS_DIR, exist_ok=True)
    os.makedirs(config.VIDEOS_PROCESSADOS_DIR, exist_ok=True)
    os.makedirs(config.FOTOS_DIR, exist_ok=True)

    ja_processados = carregar_log_processados()

    videos_pendentes = [
        f for f in os.listdir(config.VIDEOS_DIR)
        if f.lower().endswith(EXTENSOES_VIDEO) and f not in ja_processados
    ]

    if not videos_pendentes:
        print("Nenhum vídeo novo pra processar em data/videos_entrada/")
        return

    print(f"{len(videos_pendentes)} vídeo(s) pendente(s). Carregando modelos...")

    detector = Detector()
    ocr = LeitorOCR()
    classificador_mm = ClassificadorMarcaModelo()
    planilha = PlanilhaWriter()
    dedup = Deduplicador()

    for nome_video in videos_pendentes:
        caminho_video = os.path.join(config.VIDEOS_DIR, nome_video)

        processar_video(caminho_video, detector, ocr, planilha, dedup, classificador_mm=classificador_mm)

        marcar_como_processado(nome_video)
        destino = os.path.join(config.VIDEOS_PROCESSADOS_DIR, nome_video)
        shutil.move(caminho_video, destino)

    print(f"\nConcluído. Planilha atualizada em: {config.PLANILHA_PATH}")


if __name__ == "__main__":
    main()
