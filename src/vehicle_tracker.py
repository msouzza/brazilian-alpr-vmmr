"""
Rastreador de veículos para o Modo Varredura Total (Total Scan / Forensic Vehicle Search).
Acompanha veículos ao longo dos frames da câmera utilizando IoU (Intersection over Union),
mantém trilhas de veículos ativas, consolida o melhor frame (maior resolução e área),
acumula leituras e votações de placas e suprime falsos positivos / ruídos no horizonte.
"""

import os
import sys
from collections import defaultdict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.validator import desempatar_placas, desempatar_caracteres, validar_placa, obter_placa_ou_mascara


def calcular_iou(boxA: tuple, boxB: tuple) -> float:
    """Calcula a interseção sobre união (IoU) entre duas bboxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0.0


def filtrar_caixas_redundantes(boxes: list[dict], iomin_thresh: float = 0.60) -> list[dict]:
    """
    Remove caixas duplicadas ou altamente contidas (ex: detecção da cabine de um caminhão
    completamente contida dentro da detecção do caminhão inteiro).
    """
    if not boxes:
        return []

    # Ordena por área decrescente (prioriza caixas maiores e mais completas)
    boxes_ordenadas = sorted(
        boxes,
        key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]),
        reverse=True,
    )
    mantidas = []

    for b in boxes_ordenadas:
        bx1, by1, bx2, by2 = b["bbox"]
        area_b = (bx2 - bx1) * (by2 - by1)
        suprimida = False

        for k in mantidas:
            kx1, ky1, kx2, ky2 = k["bbox"]
            ix1, iy1 = max(bx1, kx1), max(by1, ky1)
            ix2, iy2 = min(bx2, kx2), min(by2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if area_b > 0 and (inter / area_b) > iomin_thresh:
                suprimida = True
                break

        if not suprimida:
            mantidas.append(b)

    return mantidas


def eh_capo_camera(bbox: tuple, frame_shape: tuple) -> bool:
    """
    Verifica se a detecção corresponde ao próprio capô/painel do veículo que grava o vídeo
    (largura > 85% do frame, na extremidade inferior e aspecto extremamente achatado).
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    if bw > 0.85 * w and y2 >= 0.90 * h and (bw / max(1, bh)) > 3.5:
        return True
    return False


class TrilhaVeiculo:
    """
    Representa uma trilha contínua de um veículo monitorado pela câmera.
    """

    def __init__(self, track_id: int, bbox_red: tuple, bbox_orig: tuple, cls_id: int, conf: float,
                 frame_orig: np.ndarray, timestamp_seg: float):
        self.track_id = track_id
        self.bbox_red = bbox_red          # bbox no frame reduzido
        self.cls_id = cls_id
        self.conf = conf
        self.frames_count = 1
        self.lost_frames = 0
        self.finalizado = False

        w_red = bbox_red[2] - bbox_red[0]
        h_red = bbox_red[3] - bbox_red[1]
        self.max_w = w_red
        self.max_h = h_red
        self.maior_area = w_red * h_red

        self.melhor_frame = frame_orig.copy()
        self.melhor_bbox_veiculo = bbox_orig
        self.melhor_t_seg = timestamp_seg
        self.ultimo_t_seg = timestamp_seg

        self.votos_tipo = defaultdict(int)
        self.votos_tipo[cls_id] += 1
        self.votos_cor = defaultdict(int)
        self.votos_marca_modelo = defaultdict(int)

        # Informações de placa associadas à trilha
        self.teve_placa = False
        self.melhor_placa = None
        self.melhor_bbox_placa = None
        self.melhor_score_placa = -1.0
        self.conf_ocr = 0.0
        self.votos_posicao = [defaultdict(int) for _ in range(7)]
        self.leituras_validas = defaultdict(int)

    def atualizar(self, bbox_red: tuple, bbox_orig: tuple, cls_id: int, conf: float,
                  frame_orig: np.ndarray, timestamp_seg: float):
        self.frames_count += 1
        self.lost_frames = 0
        self.bbox_red = bbox_red
        self.cls_id = cls_id
        self.conf = max(self.conf, conf)
        self.votos_tipo[cls_id] += 1
        self.ultimo_t_seg = timestamp_seg

        w_red = bbox_red[2] - bbox_red[0]
        h_red = bbox_red[3] - bbox_red[1]
        self.max_w = max(self.max_w, w_red)
        self.max_h = max(self.max_h, h_red)
        area = w_red * h_red

        # Se ainda não teve leitura de placa com alto score e a área atual é maior (mais nítido e próximo)
        if not self.teve_placa and area > self.maior_area:
            self.maior_area = area
            self.melhor_frame = frame_orig.copy()
            self.melhor_bbox_veiculo = bbox_orig
            self.melhor_t_seg = timestamp_seg

    def registrar_placa(self, placa_candidata: str, placa_valida: str | None,
                        conf_ocr: float, bbox_placa_orig: tuple, score_leitura: float,
                        frame_orig: np.ndarray, bbox_veiculo_orig: tuple, timestamp_seg: float):
        self.teve_placa = True

        if placa_valida:
            self.leituras_validas[placa_valida] += 1

        if len(placa_candidata) == 7:
            for pos, char in enumerate(placa_candidata):
                if char not in ("?", "-"):
                    self.votos_posicao[pos][char] += 1

        # Dá preferência a frames onde a leitura da placa teve maior pontuação
        if score_leitura > self.melhor_score_placa:
            self.melhor_score_placa = score_leitura
            self.melhor_placa = placa_candidata
            self.melhor_bbox_placa = bbox_placa_orig
            self.conf_ocr = conf_ocr
            self.melhor_frame = frame_orig.copy()
            self.melhor_bbox_veiculo = bbox_veiculo_orig
            self.melhor_t_seg = timestamp_seg

    def contem_ponto(self, px: float, py: float) -> bool:
        x1, y1, x2, y2 = self.bbox_red
        return x1 <= px <= x2 and y1 <= py <= y2

    def obter_placa_consolidada(self) -> str:
        """
        Consolida a placa definitiva da trilha a partir das leituras válidas,
        votação por posição e desempate inteligente de caracteres (Z vs L, 0 vs D, 8 vs B).
        """
        if not self.leituras_validas:
            # Se não teve leitura válida direta, tenta reconstruir por votos_posicao
            chars = []
            for pos in range(7):
                votos = self.votos_posicao[pos]
                if not votos:
                    chars.append("?")
                else:
                    max_votos = max(votos.values())
                    top_chars = [c for c, v in votos.items() if v == max_votos]
                    char_venc = top_chars[0]
                    for c_cand in top_chars[1:]:
                        char_venc = desempatar_caracteres(char_venc, c_cand, pos)
                    chars.append(char_venc)
            cand = "".join(chars)
            placa_val = validar_placa(cand)
            if placa_val:
                return placa_val
            return obter_placa_ou_mascara(cand)

        # Se houver apenas uma leitura válida
        if len(self.leituras_validas) == 1:
            return next(iter(self.leituras_validas.keys()))

        # Ordena leituras válidas por contagem de votos decrescente
        leituras_ordenadas = sorted(self.leituras_validas.items(), key=lambda x: x[1], reverse=True)
        max_votos = leituras_ordenadas[0][1]
        candidatas_top = [p for p, v in leituras_ordenadas if v == max_votos]

        if len(candidatas_top) == 1:
            return candidatas_top[0]

        # Em caso de empate entre as mais votadas (ex: 1 voto EZU9D76 e 1 voto ELU9D76):
        # Desempata usando as regras de desempate de caracteres
        placa_vencedora = candidatas_top[0]
        for p_concorrente in candidatas_top[1:]:
            placa_vencedora = desempatar_placas(placa_vencedora, p_concorrente)

        return placa_vencedora


class RastreadorVeiculos:
    """
    Rastreador leve baseado em IoU temporal para agregação de veículos e placas.
    """

    def __init__(self, iou_limiar: float = 0.20, max_frames_perdidos: int = 10):
        self.iou_limiar = iou_limiar
        self.max_frames_perdidos = max_frames_perdidos
        self.trilhas: list[TrilhaVeiculo] = []
        self.proximo_id = 1

    def atualizar_frame(self, veiculos_det: list[dict], frame_orig: np.ndarray,
                        escala: float, timestamp_seg: float) -> list[TrilhaVeiculo]:
        """
        Recebe as detecções de veículos do frame reduzido e atualiza as trilhas ativas.
        Retorna a lista de trilhas finalizadas (que saíram de cena) neste frame.
        """
        trilhas_ativas = [t for t in self.trilhas if not t.finalizado]
        indices_associados = set()

        for v in veiculos_det:
            bbox_red = v["bbox"]
            cls_id = v["cls"]
            conf = v["conf"]

            # Bbox original sem margem para medição pura
            rx1, ry1, rx2, ry2 = bbox_red
            ox1, oy1 = max(0, int(rx1 / escala)), max(0, int(ry1 / escala))
            ox2, oy2 = min(frame_orig.shape[1], int(rx2 / escala)), min(frame_orig.shape[0], int(ry2 / escala))
            bbox_orig = (ox1, oy1, ox2, oy2)

            melhor_iou = 0.0
            melhor_idx = -1

            for idx, tr in enumerate(trilhas_ativas):
                if idx not in indices_associados:
                    score = calcular_iou(bbox_red, tr.bbox_red)
                    if score > melhor_iou:
                        melhor_iou = score
                        melhor_idx = idx

            if melhor_iou >= self.iou_limiar and melhor_idx >= 0:
                trilhas_ativas[melhor_idx].atualizar(bbox_red, bbox_orig, cls_id, conf, frame_orig, timestamp_seg)
                indices_associados.add(melhor_idx)
            else:
                nova_trilha = TrilhaVeiculo(
                    self.proximo_id, bbox_red, bbox_orig, cls_id, conf, frame_orig, timestamp_seg
                )
                self.trilhas.append(nova_trilha)
                indices_associados.add(len(trilhas_ativas))
                trilhas_ativas.append(nova_trilha)
                self.proximo_id += 1

        trilhas_finalizadas = []
        for idx, tr in enumerate(trilhas_ativas):
            if idx not in indices_associados:
                tr.lost_frames += 1
                if tr.lost_frames > self.max_frames_perdidos:
                    tr.finalizado = True
                    trilhas_finalizadas.append(tr)

        return trilhas_finalizadas

    def associar_placa(self, bbox_placa_red: tuple, placa_candidata: str, placa_valida: str | None,
                       conf_ocr: float, bbox_placa_orig: tuple, score_leitura: float,
                       frame_orig: np.ndarray, escala: float, timestamp_seg: float) -> TrilhaVeiculo:
        """
        Associa uma placa detectada à trilha de veículo correspondente.
        Em casos de veículos próximos ou sobrepostos (ex: veículo estacionado com outro logo atrás),
        prioriza estritamente o veículo cujo centro esteja mais próximo da placa e cuja caixa delimite
        a placa de forma mais compacta e proporcional, evitando atribuir a placa a um veículo de fundo maior
        ou excessivamente deslocado na borda.
        """
        px1, py1, px2, py2 = bbox_placa_red
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        pw = px2 - px1
        ph = py2 - py1
        area_placa = max(1, pw * ph)

        trilhas_ativas = [t for t in self.trilhas if not t.finalizado]

        # 1. Avalia trilhas ativas que englobam ou sobrepõem significativamente a placa
        candidatos = []
        for tr in trilhas_ativas:
            vx1, vy1, vx2, vy2 = tr.bbox_red
            vcx = (vx1 + vx2) / 2.0
            vcy = (vy1 + vy2) / 2.0
            vw = max(1, vx2 - vx1)
            vh = max(1, vy2 - vy1)
            varea = vw * vh
            dist_centro = ((pcx - vcx) ** 2 + (pcy - vcy) ** 2) ** 0.5

            contem_centro = (vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2)

            ix1, iy1 = max(px1, vx1), max(py1, vy1)
            ix2, iy2 = min(px2, vx2), min(py2, vy2)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            frac_placa = inter_area / area_placa

            # Margem relativa horizontal do centro da placa em relação às bordas do veículo (0.5 = centro, 0 = borda)
            dist_borda_esq = pcx - vx1
            dist_borda_dir = vx2 - pcx
            margem_borda_x = min(dist_borda_esq, dist_borda_dir)
            rel_borda_x = margem_borda_x / vw

            if contem_centro or frac_placa >= 0.50:
                # Custo composto: penaliza distância ao centro e caixas gigantes de fundo,
                # bonifica centralidade e penaliza placas coladas na borda lateral de carro vizinho
                custo = dist_centro * (1.0 + (varea / 35000.0)) - (rel_borda_x * 50.0)
                if rel_borda_x < 0.12:
                    custo += 60.0  # Penalidade se a placa estiver colada na borda do carro vizinho

                candidatos.append((custo, tr))

        if candidatos:
            candidatos.sort(key=lambda x: x[0])
            melhor_trilha = candidatos[0][1]

            # Verificação de sucessão/separação de veículos contíguos:
            # Se a melhor trilha já possui leituras válidas confirmadas e a nova placa é flagrantemente
            # incompatível (>= 4 caracteres distintos de um carro anterior), trata-se de um novo veículo
            # estacionado contíguo que entrou no mesmo espaço visual. Finaliza a trilha anterior e cria nova.
            if placa_valida and melhor_trilha.leituras_validas:
                placa_anterior = melhor_trilha.obter_placa_consolidada()
                if len(placa_anterior) == 7 and len(placa_valida) == 7:
                    diffs = sum(1 for a, b in zip(placa_anterior, placa_valida) if a != b)
                    if diffs >= 4:
                        melhor_trilha.finalizado = True
                        rx1, ry1, rx2, ry2 = melhor_trilha.bbox_red
                        ox1, oy1 = max(0, int(rx1 / escala)), max(0, int(ry1 / escala))
                        ox2, oy2 = min(frame_orig.shape[1], int(rx2 / escala)), min(frame_orig.shape[0], int(ry2 / escala))
                        nova_trilha = TrilhaVeiculo(
                            self.proximo_id, melhor_trilha.bbox_red, (ox1, oy1, ox2, oy2),
                            melhor_trilha.cls_id, melhor_trilha.conf, frame_orig, timestamp_seg
                        )
                        nova_trilha.registrar_placa(
                            placa_candidata, placa_valida, conf_ocr, bbox_placa_orig, score_leitura,
                            frame_orig, (ox1, oy1, ox2, oy2), timestamp_seg
                        )
                        self.trilhas.append(nova_trilha)
                        self.proximo_id += 1
                        return nova_trilha

            rx1, ry1, rx2, ry2 = melhor_trilha.bbox_red
            ox1, oy1 = max(0, int(rx1 / escala)), max(0, int(ry1 / escala))
            ox2, oy2 = min(frame_orig.shape[1], int(rx2 / escala)), min(frame_orig.shape[0], int(ry2 / escala))
            melhor_trilha.registrar_placa(
                placa_candidata, placa_valida, conf_ocr, bbox_placa_orig, score_leitura,
                frame_orig, (ox1, oy1, ox2, oy2), timestamp_seg
            )
            return melhor_trilha

        # 2. Se nenhuma trilha englobar o centro, avalia trilhas próximas (distância euclidiana entre centros)
        candidatos_proximos = []
        for tr in trilhas_ativas:
            vx1, vy1, vx2, vy2 = tr.bbox_red
            vcx = (vx1 + vx2) / 2.0
            vcy = (vy1 + vy2) / 2.0
            vw = max(1, vx2 - vx1)
            vh = max(1, vy2 - vy1)
            varea = vw * vh
            dist = ((pcx - vcx) ** 2 + (pcy - vcy) ** 2) ** 0.5
            if dist < 160.0:
                custo = dist * (1.0 + (varea / 35000.0))
                candidatos_proximos.append((custo, tr))

        if candidatos_proximos:
            candidatos_proximos.sort(key=lambda x: x[0])
            melhor_tr = candidatos_proximos[0][1]
            rx1, ry1, rx2, ry2 = melhor_tr.bbox_red
            ox1, oy1 = max(0, int(rx1 / escala)), max(0, int(ry1 / escala))
            ox2, oy2 = min(frame_orig.shape[1], int(rx2 / escala)), min(frame_orig.shape[0], int(ry2 / escala))
            melhor_tr.registrar_placa(
                placa_candidata, placa_valida, conf_ocr, bbox_placa_orig, score_leitura,
                frame_orig, (ox1, oy1, ox2, oy2), timestamp_seg
            )
            return melhor_tr

        # 3. Fallback: Cria trilha geométrica ao redor da placa
        h_frame, w_frame = frame_orig.shape[:2]
        pw = px2 - px1
        ph = py2 - py1
        vx1 = max(0, int(px1 - pw * 1.5))
        vy1 = max(0, int(py1 - ph * 3.5))
        vx2 = min(int(w_frame * escala), int(px2 + pw * 1.5))
        vy2 = min(int(h_frame * escala), int(py2 + ph * 1.5))
        v_bbox_red = (vx1, vy1, vx2, vy2)

        ox1, oy1 = max(0, int(vx1 / escala)), max(0, int(vy1 / escala))
        ox2, oy2 = min(w_frame, int(vx2 / escala)), min(h_frame, int(vy2 / escala))
        v_bbox_orig = (ox1, oy1, ox2, oy2)

        nova_trilha = TrilhaVeiculo(
            self.proximo_id, v_bbox_red, v_bbox_orig, 2, 0.5, frame_orig, timestamp_seg
        )
        nova_trilha.registrar_placa(
            placa_candidata, placa_valida, conf_ocr, bbox_placa_orig, score_leitura,
            frame_orig, v_bbox_orig, timestamp_seg
        )
        self.trilhas.append(nova_trilha)
        self.proximo_id += 1
        return nova_trilha

    def finalizar_restantes(self) -> list[TrilhaVeiculo]:
        """Finaliza e retorna todas as trilhas restantes ao final do processamento do vídeo."""
        restantes = [t for t in self.trilhas if not t.finalizado]
        for t in restantes:
            t.finalizado = True
        return restantes
