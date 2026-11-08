"""
Processa um arquivo de vídeo: extrai frames, detecta veículos/placas,
roda OCR, valida, deduplica, salva foto e grava na planilha.

Suporta dois modos de operação:
1. MODO VARREDURA TOTAL (total_scan - padrão):
   Rastreia e cataloga TODOS os veículos presentes na cena (em movimento, parados
   ou estacionados), identificando Tipo, Cor, Marca/Modelo via SigLIP (Fase 2)
   e associando leituras de placa quando visíveis ou marcando como SEM_PLACA.
2. MODO PLACA APENAS (plate_only):
   Processamento estrito acionado apenas para veículos com placas detectadas.
"""

import os
import sys
import time
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
from src.vehicle_classifier import (
    classificar_veiculo,
    estimar_cor_lataria,
    ClassificadorMarcaModelo,
    MAPA_TIPOS_VEICULO,
)
from src.vehicle_tracker import (
    RastreadorVeiculos,
    TrilhaVeiculo,
    eh_capo_camera,
    filtrar_caixas_redundantes,
)
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


def _bbox_para_original(bbox, escala, shape_original, margem=None):
    """Converte a bbox do frame reduzido pro frame original, com uma pequena margem."""
    if margem is None:
        margem = 0.0 if getattr(config, "OCR_ENGINE", "") == "fast_plate_ocr" else 0.08
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


def _consolidar_e_registrar_trilha(
    tr: TrilhaVeiculo,
    detector: Detector,
    planilha: PlanilhaWriter,
    dedup: Deduplicador,
    classificador_mm: ClassificadorMarcaModelo,
    data_gravacao: datetime,
    nome_video: str,
) -> bool:
    """
    Consolida os atributos de uma trilha de veículo finalizada e registra na planilha e em foto.
    Aplica filtros de tamanho e persistência para eliminar ruídos no horizonte e falsos positivos.
    Retorna True se o veículo foi registrado, False se descartado.
    """
    tem_placa_valida = bool(tr.leituras_validas)

    # 1. Filtro de persistência e tamanho mínimo (veículos com placa válida são sempre preservados)
    if not tem_placa_valida:
        if tr.frames_count < config.MIN_VEHICLE_FRAMES:
            tr.melhor_frame = None
            return False
        if tr.max_w < config.MIN_VEHICLE_WIDTH or tr.max_h < config.MIN_VEHICLE_HEIGHT:
            tr.melhor_frame = None
            return False

    # 2. Definição da Placa
    placa = tr.obter_placa_consolidada()
    if placa and placa not in ("SEM_PLACA", "???????", "???-????"):
        conf_ocr = tr.conf_ocr
        bbox_placa = tr.melhor_bbox_placa
    else:
        placa = "SEM_PLACA"
        conf_ocr = 0.0
        bbox_placa = None

    # 3. Consolidação do Tipo
    if tr.votos_tipo:
        cls_venc = max(tr.votos_tipo.items(), key=lambda x: x[1])[0]
        tipo = MAPA_TIPOS_VEICULO.get(cls_venc, "Carro")
    else:
        tipo = "Carro"

    # 4. Estimativa da Cor da Lataria no melhor recorte do veículo
    crop_melhor_veiculo = detector.recortar(tr.melhor_frame, tr.melhor_bbox_veiculo)
    if crop_melhor_veiculo.size > 0:
        cor = estimar_cor_lataria(crop_melhor_veiculo)
    else:
        cor = "Indefinida"

    # 5. SigLIP VMMR: executado APENAS UMA VEZ por veículo no melhor frame
    marca_modelo = "-"
    top3_resumo = "-"
    confiabilidade_vmmr = "Baixa / Ambígua"

    if classificador_mm is not None and crop_melhor_veiculo.size > 0:
        res_mm = classificador_mm.classificar(crop_melhor_veiculo, tipo)
        marca_modelo = res_mm.get("marca_modelo", "-")
        top3_resumo = res_mm.get("top3", "-")
        confiabilidade_vmmr = res_mm.get("confiabilidade", "Baixa / Ambígua")
        if top3_resumo and top3_resumo != "-":
            print(f"  [SigLIP VMMR] Trilha #{tr.track_id} ({tipo}): Top-3: {top3_resumo} | Confiabilidade: {confiabilidade_vmmr}")

    # 6. Separação de Top-3 nas 3 descobertas individuais
    desc_1 = "-"
    desc_2 = "-"
    desc_3 = "-"
    if top3_resumo and top3_resumo != "-":
        partes = [p.strip() for p in top3_resumo.split(",") if p.strip()]
        if len(partes) > 0:
            desc_1 = partes[0]
        if len(partes) > 1:
            desc_2 = partes[1]
        if len(partes) > 2:
            desc_3 = partes[2]

    # 7. Deduplicação
    if not dedup.deve_registrar(placa, tr.melhor_t_seg):
        print(f"  [DEDUP] Veículo ignorado por deduplicação: {placa}")
        tr.melhor_frame = None
        return False

    # 8. Salvamento da Foto Forense
    info_veiculo_final = {"tipo": tipo, "cor": cor, "marca_modelo": marca_modelo}
    dt_deteccao = data_gravacao + timedelta(seconds=tr.melhor_t_seg)

    caminho_foto = salvar_foto_veiculo(
        tr.melhor_frame,
        placa,
        dt_deteccao,
        bbox=bbox_placa,
        bbox_veiculo=tr.melhor_bbox_veiculo,
        info_veiculo=info_veiculo_final,
    )

    # 9. Gravação na Planilha
    planilha.registrar(
        placa=placa,
        tipo=tipo,
        cor=cor,
        marca_modelo=marca_modelo,
        dt=dt_deteccao,
        confianca_ocr=conf_ocr,
        caminho_foto=caminho_foto,
        video_origem=nome_video,
        desc_1=desc_1,
        desc_2=desc_2,
        desc_3=desc_3,
        confiabilidade=confiabilidade_vmmr,
    )

    tr.melhor_frame = None  # Libera memória imediatamente
    print(f"  [OK] Veículo consolidado (Trilha #{tr.track_id}): {placa} | {tipo} | {marca_modelo} (Confiabilidade: {confiabilidade_vmmr}) | Cor: {cor} (conf OCR: {conf_ocr:.2f}) -> {caminho_foto}")
    return True


def _processar_video_total_scan(
    caminho_video: str,
    detector: Detector,
    ocr: LeitorOCR,
    planilha: PlanilhaWriter,
    dedup: Deduplicador,
    data_gravacao: datetime | None = None,
    classificador_mm: ClassificadorMarcaModelo | None = None,
):
    """
    Executa a Varredura Total Forense: detecta e rastreia todos os veículos da cena
    (em movimento ou estacionados/parados), identificando Tipo, Cor, Marca/Modelo
    via SigLIP e OCR nas placas visíveis.
    """
    if classificador_mm is None:
        classificador_mm = ClassificadorMarcaModelo()

    t_inicio = time.time()
    nome_video = os.path.basename(caminho_video)
    print(f"\n[VideoProcessor] [MODO VARREDURA TOTAL] Processando: {nome_video}")

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

    rastreador = RastreadorVeiculos(iou_limiar=0.20, max_frames_perdidos=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % config.FRAME_SKIP == 0:
            if total_frames > 0 and (frame_idx % (config.FRAME_SKIP * 15) == 0):
                pct = (frame_idx / total_frames) * 100
                print(f"[VideoProcessor] Progresso: {pct:.1f}% (Frame {frame_idx}/{total_frames})")

            frame_reduzido, escala = _redimensionar(frame)
            timestamp_seg = frame_idx / fps

            # 1. Detecção de veículos no frame
            veiculos_raw = detector.detectar_veiculos(frame_reduzido)
            # Remove capô/painel do carro que grava o vídeo
            veiculos_filtrados = [
                v for v in veiculos_raw
                if not eh_capo_camera(v["bbox"], frame_reduzido.shape)
            ]
            # Suprime caixas contidas/redundantes (ex: cabine vs caminhão inteiro)
            veiculos_det = filtrar_caixas_redundantes(veiculos_filtrados)

            # Atualiza o rastreador temporal
            trilhas_finalizadas = rastreador.atualizar_frame(veiculos_det, frame, escala, timestamp_seg)

            # 2. Detecção de placas no frame
            deteccoes = detector.detectar(frame_reduzido)
            if deteccoes:
                for det in deteccoes:
                    bbox_orig = _bbox_para_original(det["bbox"], escala, frame.shape)
                    crop = detector.recortar(frame, bbox_orig)
                    leituras = ocr.ler(crop)

                    placa_final = None
                    placa_valida = None
                    confianca_final = 0.0

                    for texto, conf_ocr in _candidatos_placa(leituras):
                        pv = validar_placa(texto)
                        if pv:
                            placa_valida = pv
                            placa_final = pv
                            confianca_final = conf_ocr
                            break

                    if not placa_final and leituras:
                        melhor_texto, melhor_conf = max(leituras, key=lambda x: x[1])
                        placa_final = obter_placa_ou_mascara(melhor_texto)
                        confianca_final = melhor_conf

                    if not placa_final:
                        placa_final = "???-????"
                        confianca_final = 0.0

                    score_atual = _calcular_score_leitura(placa_final, confianca_final, bbox_orig)

                    rastreador.associar_placa(
                        det["bbox"],
                        placa_final,
                        placa_valida,
                        confianca_final,
                        bbox_orig,
                        score_atual,
                        frame,
                        escala,
                        timestamp_seg,
                    )

            # Processa veículos que saíram de cena durante o vídeo
            for tr in trilhas_finalizadas:
                if _consolidar_e_registrar_trilha(
                    tr, detector, planilha, dedup, classificador_mm, data_gravacao, nome_video
                ):
                    total_deteccoes += 1

        frame_idx += 1

    cap.release()

    # Finaliza e consolida todas as trilhas restantes ao final do vídeo
    trilhas_restantes = rastreador.finalizar_restantes()
    for tr in trilhas_restantes:
        if _consolidar_e_registrar_trilha(
            tr, detector, planilha, dedup, classificador_mm, data_gravacao, nome_video
        ):
            total_deteccoes += 1

    tempo_total = time.time() - t_inicio
    fps_proc = total_frames / tempo_total if tempo_total > 0 else 0.0

    print(f"[VideoProcessor] Finalizado. {total_deteccoes} veículo(s) registrado(s) em {nome_video}")
    print(f"[VideoProcessor] Tempo total de processamento: {tempo_total:.1f} segundos ({fps_proc:.1f} fps)")


def _processar_video_plate_only(
    caminho_video: str,
    detector: Detector,
    ocr: LeitorOCR,
    planilha: PlanilhaWriter,
    dedup: Deduplicador,
    data_gravacao: datetime | None = None,
    classificador_mm: ClassificadorMarcaModelo | None = None,
):
    """
    Modo estrito clássico: processa apenas veículos que possuem placa detectada.
    """
    if classificador_mm is None:
        classificador_mm = ClassificadorMarcaModelo()

    t_inicio = time.time()
    nome_video = os.path.basename(caminho_video)
    print(f"\n[VideoProcessor] [MODO PLACA APENAS] Processando: {nome_video}")

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

            if deteccoes:
                veiculos_det = detector.detectar_veiculos(frame_reduzido)

                for det in deteccoes:
                    bbox_orig = _bbox_para_original(det["bbox"], escala, frame.shape)
                    crop = detector.recortar(frame, bbox_orig)
                    leituras = ocr.ler(crop)

                    veiculo_assoc = detector.associar_veiculo_placa(det["bbox"], veiculos_det, frame_reduzido.shape)
                    v_bbox_orig = _bbox_para_original(veiculo_assoc["bbox"], escala, frame.shape, margem=0.0)
                    crop_veiculo = detector.recortar(frame, v_bbox_orig)
                    info_veiculo = classificar_veiculo(crop_veiculo, veiculo_assoc["cls"], classificador_mm=None)

                    placa_final = None
                    confianca_final = 0.0

                    for texto, conf_ocr in _candidatos_placa(leituras):
                        placa_valida = validar_placa(texto)
                        if placa_valida:
                            placa_final = placa_valida
                            confianca_final = conf_ocr
                            break

                    if not placa_final and leituras:
                        melhor_texto, melhor_conf = max(leituras, key=lambda x: x[1])
                        placa_final = obter_placa_ou_mascara(melhor_texto)
                        confianca_final = melhor_conf

                    if not placa_final:
                        placa_final = "???-????"
                        confianca_final = 0.0

                    timestamp_seg = frame_idx / fps
                    score_atual = _calcular_score_leitura(placa_final, confianca_final, bbox_orig)

                    evento_encontrado = None
                    for ev in eventos_veiculos:
                        if (timestamp_seg - ev["ultimo_t_seg"]) <= config.DEDUP_WINDOW_SECONDS:
                            evento_encontrado = ev
                            break

                    if evento_encontrado is not None:
                        evento_encontrado["ultimo_t_seg"] = timestamp_seg
                        if len(placa_final) == 7:
                            for pos_idx, char in enumerate(placa_final):
                                if char not in ("?", "-"):
                                    evento_encontrado["votos_posicao"][pos_idx][char] += 1

                        if placa_valida:
                            evento_encontrado["leituras_validas"][placa_valida] += 1

                        evento_encontrado["votos_tipo"][info_veiculo["tipo"]] += 1
                        evento_encontrado["votos_cor"][info_veiculo["cor"]] += 1
                        if info_veiculo["marca_modelo"] != "-":
                            evento_encontrado["votos_marca_modelo"][info_veiculo["marca_modelo"]] += 1

                        if score_atual > evento_encontrado["melhor_score"]:
                            evento_encontrado["melhor_score"] = score_atual
                            evento_encontrado["melhor_placa"] = placa_final
                            evento_encontrado["melhor_frame"] = frame.copy()
                            evento_encontrado["melhor_bbox"] = bbox_orig
                            evento_encontrado["melhor_bbox_veiculo"] = v_bbox_orig
                            evento_encontrado["conf_ocr"] = confianca_final
                            evento_encontrado["t_seg"] = timestamp_seg
                    else:
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

    for ev in eventos_veiculos:
        if ev["leituras_validas"]:
            placa = max(ev["leituras_validas"].items(), key=lambda x: x[1])[0]
        else:
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

        if ev.get("votos_tipo"):
            tipo = max(ev["votos_tipo"].items(), key=lambda x: x[1])[0]
        else:
            tipo = "Carro"

        if ev.get("votos_cor"):
            cores_validas = {c: count for c, count in ev["votos_cor"].items() if c != "Indefinida"}
            if cores_validas:
                cor = max(cores_validas.items(), key=lambda x: x[1])[0]
            else:
                cor = max(ev["votos_cor"].items(), key=lambda x: x[1])[0]
        else:
            cor = "Indefinida"

        marca_modelo = "-"
        top3_resumo = "-"
        confiabilidade_vmmr = "Baixa / Ambígua"

        if classificador_mm is not None and ev.get("melhor_bbox_veiculo") is not None:
            crop_melhor_veiculo = detector.recortar(ev["melhor_frame"], ev["melhor_bbox_veiculo"])
            res_mm = classificador_mm.classificar(crop_melhor_veiculo, tipo)
            marca_modelo = res_mm.get("marca_modelo", "-")
            top3_resumo = res_mm.get("top3", "-")
            confiabilidade_vmmr = res_mm.get("confiabilidade", "Baixa / Ambígua")
            if top3_resumo and top3_resumo != "-":
                print(f"  [SigLIP VMMR] Top-3 candidatos: {top3_resumo}")
                print(f"  [SigLIP VMMR] Confiabilidade: {confiabilidade_vmmr}")
        elif ev.get("votos_marca_modelo"):
            marca_modelo = max(ev["votos_marca_modelo"].items(), key=lambda x: x[1])[0]

        desc_1 = "-"
        desc_2 = "-"
        desc_3 = "-"
        if top3_resumo and top3_resumo != "-":
            partes = [p.strip() for p in top3_resumo.split(",") if p.strip()]
            if len(partes) > 0:
                desc_1 = partes[0]
            if len(partes) > 1:
                desc_2 = partes[1]
            if len(partes) > 2:
                desc_3 = partes[2]

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
        planilha.registrar(
            placa=placa,
            tipo=tipo,
            cor=cor,
            marca_modelo=marca_modelo,
            dt=dt_deteccao,
            confianca_ocr=conf_ocr,
            caminho_foto=caminho_foto,
            video_origem=nome_video,
            desc_1=desc_1,
            desc_2=desc_2,
            desc_3=desc_3,
            confiabilidade=confiabilidade_vmmr,
        )

        total_deteccoes += 1
        print(f"  [OK] Veículo consolidado: {placa} | {tipo} | {marca_modelo} (Confiabilidade: {confiabilidade_vmmr}) | Cor: {cor} (conf OCR: {conf_ocr:.2f}) -> {caminho_foto}")

    tempo_total = time.time() - t_inicio
    fps_proc = total_frames / tempo_total if tempo_total > 0 else 0.0

    print(f"[VideoProcessor] Finalizado. {total_deteccoes} veículo(s) registrado(s) em {nome_video}")
    print(f"[VideoProcessor] Tempo total de processamento: {tempo_total:.1f} segundos ({fps_proc:.1f} fps)")


def processar_video(
    caminho_video: str,
    detector: Detector,
    ocr: LeitorOCR,
    planilha: PlanilhaWriter,
    dedup: Deduplicador,
    data_gravacao: datetime | None = None,
    classificador_mm: ClassificadorMarcaModelo | None = None,
):
    """
    Despacha o processamento do vídeo de acordo com o OPERATION_MODE definido em config.py.
    """
    modo = getattr(config, "OPERATION_MODE", "total_scan")
    if modo == "plate_only":
        return _processar_video_plate_only(
            caminho_video, detector, ocr, planilha, dedup, data_gravacao, classificador_mm
        )
    return _processar_video_total_scan(
        caminho_video, detector, ocr, planilha, dedup, data_gravacao, classificador_mm
    )
