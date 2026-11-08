"""
Evita gravar a mesma placa repetidamente enquanto o veículo aparece em
vários frames seguidos do vídeo.
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Deduplicador:
    def __init__(self):
        self._ultima_deteccao = {}  # {placa: timestamp}

    def deve_registrar(self, placa: str, timestamp_video: float) -> bool:
        """
        timestamp_video: tempo (em segundos) dentro do próprio vídeo sendo processado,
        usado pra comparar a janela de deduplicação de forma consistente mesmo
        processando vídeo gravado (não em tempo real).
        """
        # Veículos sem placa já são individualizados pelo rastreador (trilha única)
        if placa in ("SEM_PLACA", "NÃO IDENTIFICADA", "???-????") or placa.startswith("SEM_PLACA"):
            return True

        ultima = self._ultima_deteccao.get(placa)

        if ultima is None or (timestamp_video - ultima) > config.DEDUP_WINDOW_SECONDS:
            self._ultima_deteccao[placa] = timestamp_video
            return True

        return False
