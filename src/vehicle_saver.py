"""
Salva a foto do veículo (frame completo ou crop do veículo) associada a
uma placa detectada, com nome de arquivo padronizado.
"""

import os
import sys
import re
import cv2
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def sanitizar_nome_arquivo(nome: str) -> str:
    """Substitui caracteres proibidos em sistemas de arquivos (especialmente Windows) por sublinhado."""
    return re.sub(r'[\\/*?:"<>|]', "_", nome)


def salvar_foto_veiculo(frame, placa: str, dt: datetime, bbox: tuple | None = None,
                        bbox_veiculo: tuple | None = None, info_veiculo: dict | None = None) -> str:
    """
    Salva o frame com destaque no veículo e na placa detectada, além de etiqueta informativa.
    Retorna o caminho completo do arquivo salvo.
    """
    os.makedirs(config.FOTOS_DIR, exist_ok=True)

    foto = frame.copy()

    # 1. Desenha a caixa delimitadora do veículo se fornecida
    if bbox_veiculo is not None:
        vx1, vy1, vx2, vy2 = bbox_veiculo
        cv2.rectangle(foto, (vx1, vy1), (vx2, vy2), (255, 180, 0), 2)  # Ciano/Azul claro

        # Etiqueta do veículo
        if info_veiculo:
            tipo = info_veiculo.get("tipo", "Veículo")
            cor = info_veiculo.get("cor", "")
            marca_mod = info_veiculo.get("marca_modelo", "-")
            rotulo_veiculo = f"{tipo} {cor}"
            if marca_mod != "-":
                rotulo_veiculo += f" ({marca_mod})"

            tam_v = cv2.getTextSize(rotulo_veiculo, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            v_ly1 = max(0, vy1 - tam_v[1] - 8)
            cv2.rectangle(foto, (vx1, v_ly1), (vx1 + tam_v[0] + 10, vy1), (0, 0, 0), -1)
            cv2.putText(foto, rotulo_veiculo, (vx1 + 5, vy1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 0), 2, cv2.LINE_AA)

    # 2. Desenha a caixa delimitadora da placa e etiqueta
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Retângulo verde ao redor da placa
        cv2.rectangle(foto, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Etiqueta de texto acima da placa
        texto_label = f"Placa: {placa}"
        tamanho_texto = cv2.getTextSize(texto_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        label_y1 = max(0, y1 - tamanho_texto[1] - 10)
        label_y2 = y1
        label_x2 = min(foto.shape[1], x1 + tamanho_texto[0] + 10)

        # Fundo escuro do texto para alto contraste
        cv2.rectangle(foto, (x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
        cv2.putText(
            foto,
            texto_label,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    timestamp_str = dt.strftime("%Y-%m-%d_%H-%M-%S")
    placa_sanitizada = sanitizar_nome_arquivo(placa)
    nome_arquivo = f"{placa_sanitizada}_{timestamp_str}.jpg"
    caminho_completo = os.path.join(config.FOTOS_DIR, nome_arquivo)

    cv2.imwrite(caminho_completo, foto)

    return caminho_completo
