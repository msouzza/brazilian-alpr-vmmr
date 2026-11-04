"""
Processa um arquivo de vídeo: extrai frames, detecta veículos/placas,
roda OCR, valida, deduplica, salva foto e grava na planilha.

Detalhe importante: a DETECÇÃO roda num frame reduzido (rápido), mas o
RECORTE da placa é feito no frame ORIGINAL (4K), pra preservar os pixels
que o OCR precisa pra ler a placa.
"""

import os
import sys
import cv2
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.plate_detector import Detector
from src.ocr_reader import LeitorOCR
from src.validator import validar_placa, obter_placa_ou_mascara
from src.deduplicator import Deduplicador
from src.vehicle_saver import salvar_foto_veiculo
from src.spreadsheet_writer import PlanilhaWriter
from src.vehicle_classifier import classificar_veiculo, ClassificadorMarcaModelo
from src.utils import extrair_data_gravacao


def _redimensionar(frame):
    """Retorna (frame_reduzido, escala) onde escala = largura_reduzida / largura_original."""
    if config.RESIZE_WIDTH is None:
        return frame, 1.0
    h, w = frame.shape[:2]
    if w <= config.RESIZE_WIDTH:
        return frame, 1.0
    escala = config.RESIZE_WIDTH / w
    novo_tamanho = (config.RESIZE_WIDTH, int(h * escala))
    return cv2.resize(frame, novo_tamanho), escala


def _bbox_para_original(bbox, escala, shape_original, margem=0.08):
    """Converte a bbox do frame reduzido pro frame original, com uma pequena margem."""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = x1 / escala, y1 / escala, x2 / escala, y2 / escala

    h, w = shape_original[:2]
    mx, my = (x2 - x1) * margem, (y2 - y1) * margem
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return (x1, y1, x2, y2)


def _candidatos_placa(leituras):
    """
    O OCR às vezes divide a placa em pedaços (ex: 'ABC' + '1D23').
    Testamos cada pedaço isolado e também todos os pedaços concatenados.
    Retorna lista de (texto, confianca).
    """
    candidatos = list(leituras)
    if len(leituras) > 1:
        texto_junto = "".join(t for t, _ in leituras)
        conf_media = sum(c for _, c in leituras) / len(leituras)
        candidatos.append((texto_junto, conf_media))
    return candidatos


def _calcular_score_leitura(placa: str, conf_ocr: float, bbox: tuple) -> float:
    """
    Calcula pontuação de qualidade da leitura:
    - Placas completas sem '?' ganham pontuação máxima.
    - Menos '?' = maior prioridade.
    - Maior área da placa (carro mais próximo) = maior nitidez.
    """
    chars_conhecidos = sum(1 for c in placa if c not in ("?", "-"))
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    score = (chars_conhecidos * 100.0) + (conf_ocr * 20.0) + (area / 1000.0)
    # Bônus para formato perfeitamente válido (sem ?)
    if "?" not in placa and len(placa) == 7:
        score += 500.0

    return score


def processar_video(caminho_video: str, detector: Detector, ocr: LeitorOCR,
                     planilha: PlanilhaWriter, dedup: Deduplicador,
                     data_gravacao: datetime | None = None,
                     classificador_mm: ClassificadorMarcaModelo | None = None):
    """
    Processa o vídeo usando rastreamento e seleção do melhor frame (Best-Frame Selection).
    Consolida as melhores leituras de cada veículo ao longo de sua passagem.
    """
    if classificador_mm is None:
        classificador_mm = ClassificadorMarcaModelo()

    nome_video = os.path.basename(caminho_video)
    print(f"\n[VideoProcessor] Processando: {nome_video}")

    if data_gravacao is None:
        data_gravacao = extrair_data_gravacao(caminho_video)
    print(f"[VideoProcessor] Início da gravação considerado: {data_gravacao}")

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"[VideoProcessor] ERRO: não consegui abrir o vídeo {caminho_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    total_deteccoes = 0

    print(f"[VideoProcessor] Total de frames: {total_frames} (~{total_frames/fps:.1f}s de vídeo)")

    # Estrutura de eventos de veículos:
    # [{'melhor_placa', 'melhor_score', 'melhor_frame', 'melhor_bbox', 'conf_ocr', 't_seg', 'ultimo_t_seg'}]
    eventos_veiculos = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % config.FRAME_SKIP == 0:
            if total_frames > 0 and (frame_idx % (config.FRAME_SKIP * 15) == 0):
                pct = (frame_idx / total_frames) * 100
                print(f"[VideoProcessor] Progresso: {pct:.1f}% (Frame {frame_idx}/{total_frames})")

            frame_reduzido, escala = _redimensionar(frame)
            deteccoes = detector.detectar(frame_reduzido)
            veiculos_det = detector.detectar_veiculos(frame_reduzido)

            for det in deteccoes:
                # Recorte no frame original (4K nativo)
                bbox_orig = _bbox_para_original(det["bbox"], escala, frame.shape)
                crop = detector.recortar(frame, bbox_orig)
                leituras = ocr.ler(crop)

                # Associação com veículo e classificação de atributos (Fase 2)
                veiculo_assoc = detector.associar_veiculo_placa(det["bbox"], veiculos_det, frame_reduzido.shape)
                v_bbox_orig = _bbox_para_original(veiculo_assoc["bbox"], escala, frame.shape, margem=0.0)
                crop_veiculo = detector.recortar(frame, v_bbox_orig)
                info_veiculo = classificar_veiculo(crop_veiculo, veiculo_assoc["cls"], classificador_mm=classificador_mm)

                placa_final = None
                confianca_final = 0.0

                # 1. Tenta validação completa direta
                for texto, conf_ocr in _candidatos_placa(leituras):
                    placa_valida = validar_placa(texto)
                    if placa_valida:
                        placa_final = placa_valida
                        confianca_final = conf_ocr
                        break

                # 2. Aplica máscara com '?' no melhor candidato
                if not placa_final and leituras:
                    melhor_texto, melhor_conf = max(leituras, key=lambda x: x[1])
                    placa_final = obter_placa_ou_mascara(melhor_texto)
                    confianca_final = melhor_conf

                # 3. Fallback com interrogações se o OCR não leu
                if not placa_final:
                    placa_final = "???-????"
                    confianca_final = 0.0

                timestamp_seg = frame_idx / fps
                score_atual = _calcular_score_leitura(placa_final, confianca_final, bbox_orig)

                # Associa a um evento existente (mesmo veículo dentro da janela de tempo)
                evento_encontrado = None
                for ev in eventos_veiculos:
                    if (timestamp_seg - ev["ultimo_t_seg"]) <= config.DEDUP_WINDOW_SECONDS:
                        evento_encontrado = ev
                        break

                if evento_encontrado is not None:
                    evento_encontrado["ultimo_t_seg"] = timestamp_seg
                    # Acumula votos nas posições de caracteres
                    if len(placa_final) == 7:
                        for pos_idx, char in enumerate(placa_final):
                            if char not in ("?", "-"):
                                evento_encontrado["votos_posicao"][pos_idx][char] += 1

                    if placa_valida:
                        evento_encontrado["leituras_validas"][placa_valida] += 1

                    # Acumula atributos do veículo
                    evento_encontrado["votos_tipo"][info_veiculo["tipo"]] += 1
                    evento_encontrado["votos_cor"][info_veiculo["cor"]] += 1
                    if info_veiculo["marca_modelo"] != "-":
                        evento_encontrado["votos_marca_modelo"][info_veiculo["marca_modelo"]] += 1

                    # Atualiza o frame visual se o atual tiver melhor resolução/proximidade
                    if score_atual > evento_encontrado["melhor_score"]:
                        evento_encontrado["melhor_score"] = score_atual
                        evento_encontrado["melhor_placa"] = placa_final
                        evento_encontrado["melhor_frame"] = frame.copy()
                        evento_encontrado["melhor_bbox"] = bbox_orig
                        evento_encontrado["melhor_bbox_veiculo"] = v_bbox_orig
                        evento_encontrado["conf_ocr"] = confianca_final
                        evento_encontrado["t_seg"] = timestamp_seg
                else:
                    # Novo veículo detectado
                    votos = [defaultdict(int) for _ in range(7)]
                    if len(placa_final) == 7:
                        for pos_idx, char in enumerate(placa_final):
                            if char not in ("?", "-"):
                                votos[pos_idx][char] += 1

                    validas = defaultdict(int)
                    if placa_valida:
                        validas[placa_valida] += 1

                    votos_tipo = defaultdict(int)
                    votos_tipo[info_veiculo["tipo"]] += 1

                    votos_cor = defaultdict(int)
                    votos_cor[info_veiculo["cor"]] += 1

                    votos_marca_modelo = defaultdict(int)
                    if info_veiculo["marca_modelo"] != "-":
                        votos_marca_modelo[info_veiculo["marca_modelo"]] += 1

                    eventos_veiculos.append({
                        "melhor_placa": placa_final,
                        "melhor_score": score_atual,
                        "melhor_frame": frame.copy(),
                        "melhor_bbox": bbox_orig,
                        "melhor_bbox_veiculo": v_bbox_orig,
                        "conf_ocr": confianca_final,
                        "t_seg": timestamp_seg,
                        "ultimo_t_seg": timestamp_seg,
                        "votos_posicao": votos,
                        "leituras_validas": validas,
                        "votos_tipo": votos_tipo,
                        "votos_cor": votos_cor,
                        "votos_marca_modelo": votos_marca_modelo,
                    })

        frame_idx += 1

    cap.release()

    # Consolidação: aplica votação temporal de colunas e salva o melhor frame
    for ev in eventos_veiculos:
        # Se houve leitura 100% válida em qualquer frame (ex: NXZ9F22), prioriza
        if ev["leituras_validas"]:
            placa = max(ev["leituras_validas"].items(), key=lambda x: x[1])[0]
        else:
            # Reconstrói a placa através da votação majoritária por coluna
            chars_votados = []
            for col in range(7):
                votos_col = ev["votos_posicao"][col]
                if votos_col:
                    char_vencedor = max(votos_col.items(), key=lambda x: x[1])[0]
                    chars_votados.append(char_vencedor)
                else:
                    chars_votados.append("?")
            candidato_votado = "".join(chars_votados)
            placa = obter_placa_ou_mascara(candidato_votado)

        # Consolida Tipo do Veículo
        if ev.get("votos_tipo"):
            tipo = max(ev["votos_tipo"].items(), key=lambda x: x[1])[0]
        else:
            tipo = "Carro"

        # Consolida Cor da Lataria (desconsidera 'Indefinida' se houver outra cor votada)
        if ev.get("votos_cor"):
            cores_validas = {c: count for c, count in ev["votos_cor"].items() if c != "Indefinida"}
            if cores_validas:
                cor = max(cores_validas.items(), key=lambda x: x[1])[0]
            else:
                cor = max(ev["votos_cor"].items(), key=lambda x: x[1])[0]
        else:
            cor = "Indefinida"

        # Consolida Marca e Modelo (ou Categoria de Carroceria)
        if ev.get("votos_marca_modelo"):
            marca_modelo = max(ev["votos_marca_modelo"].items(), key=lambda x: x[1])[0]
        else:
            marca_modelo = "-"

        info_veiculo_final = {"tipo": tipo, "cor": cor, "marca_modelo": marca_modelo}

        conf_ocr = ev["conf_ocr"]
        t_seg = ev["t_seg"]
        dt_deteccao = data_gravacao + timedelta(seconds=t_seg)

        caminho_foto = salvar_foto_veiculo(
            ev["melhor_frame"],
            placa,
            dt_deteccao,
            bbox=ev["melhor_bbox"],
            bbox_veiculo=ev.get("melhor_bbox_veiculo"),
            info_veiculo=info_veiculo_final,
        )
        planilha.registrar(placa, tipo, cor, marca_modelo, dt_deteccao, conf_ocr, caminho_foto, nome_video)

        total_deteccoes += 1
        print(f"  [OK] Veículo consolidado: {placa} | {tipo} {cor} (conf OCR: {conf_ocr:.2f}) -> {caminho_foto}")

    print(f"[VideoProcessor] Finalizado. {total_deteccoes} veículo(s) registrado(s) em {nome_video}")
