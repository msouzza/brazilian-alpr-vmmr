"""
Funções auxiliares: extração da data/hora de gravação a partir do nome do
arquivo de vídeo da câmera.
"""

import os
import re
from datetime import datetime

# Padrões comuns de nome de arquivo de dashcam. Ajuste/adicione conforme o
# nome real dos arquivos da sua DDPAI.
_PADROES = [
    # 20261104_153012  |  20261104-153012  |  20261104153012
    (re.compile(r"(\d{4})(\d{2})(\d{2})[_\-]?(\d{2})(\d{2})(\d{2})"), None),
    # 2026-11-04_15-30-12  |  2026-11-04 15:30:12
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[_ T](\d{2})[-:](\d{2})[-:](\d{2})"), None),
]


def extrair_data_gravacao(caminho_video: str) -> datetime:
    """
    Tenta extrair a data/hora do nome do arquivo. Se não conseguir,
    usa a data de modificação do arquivo (aproximação do fim da gravação).
    """
    nome = os.path.basename(caminho_video)

    for regex, _ in _PADROES:
        m = regex.search(nome)
        if not m:
            continue
        try:
            ano, mes, dia, hh, mm, ss = map(int, m.groups())
            return datetime(ano, mes, dia, hh, mm, ss)
        except ValueError:
            continue

    print(f"[Utils] AVISO: não reconheci data no nome '{nome}'. "
          f"Usando data de modificação do arquivo.")
    return datetime.fromtimestamp(os.path.getmtime(caminho_video))


if __name__ == "__main__":
    for exemplo in ["20261104_153012_0001_F.mp4", "2026-11-04_15-30-12.mp4", "2026110415301200.mp4"]:
        m = None
        for regex, _ in _PADROES:
            m = regex.search(exemplo)
            if m:
                break
        print(f"{exemplo:32} -> {m.groups() if m else None}")
