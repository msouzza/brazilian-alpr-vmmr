"""
Testes unitários de lógica pura para o rastreador de veículos e matriz de confiabilidade VMMR.
Não carrega pesos de YOLO nem SigLIP.
"""

import numpy as np
import pytest

from src.vehicle_classifier import (
    calcular_matriz_confiabilidade,
    estimar_cor_lataria,
    formatar_nome_veiculo,
    inferir_categoria_por_modelo,
)
from src.vehicle_tracker import (
    calcular_iou,
    eh_capo_camera,
    filtrar_caixas_redundantes,
)


# ---------------------------------------------------------------------------
# 1. calcular_iou
# ---------------------------------------------------------------------------

def test_calcular_iou_sobreposicao_total():
    box = (10, 20, 110, 220)
    assert calcular_iou(box, box) == pytest.approx(1.0)


def test_calcular_iou_parcial():
    box_a = (0, 0, 100, 100)
    box_b = (50, 50, 150, 150)
    # Interseção = 50*50 = 2500; União = 10000 + 10000 - 2500 = 17500
    assert calcular_iou(box_a, box_b) == pytest.approx(2500 / 17500)


def test_calcular_iou_disjuntas_e_borda():
    assert calcular_iou((0, 0, 50, 50), (60, 60, 120, 120)) == pytest.approx(0.0)
    assert calcular_iou((0, 0, 50, 50), (50, 0, 100, 50)) == pytest.approx(0.0)


def test_calcular_iou_area_zero():
    assert calcular_iou((10, 10, 10, 10), (10, 10, 10, 10)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. eh_capo_camera
# ---------------------------------------------------------------------------

def test_eh_capo_camera_detecta_rodape_central():
    frame_shape = (1080, 1920, 3)
    # Largura = 1800 (> 85% de 1920 = 1632), y2 = 1070 (>= 90% de 1080 = 972), aspecto = 1800/170 > 3.5
    bbox_capo = (60, 900, 1860, 1070)
    assert eh_capo_camera(bbox_capo, frame_shape) is True


@pytest.mark.parametrize(
    "bbox",
    [
        (400, 300, 720, 560),    # Veículo real na pista
        (60, 200, 1860, 400),    # Larga e achatada, mas no topo (y2 < 0.90*h)
        (60, 300, 1860, 1070),   # Larga e no rodapé, mas alta (aspecto < 3.5)
        (800, 950, 1200, 1070),  # No rodapé, mas estreita (bw < 0.85*w)
    ],
)
def test_eh_capo_camera_nao_descarta_veiculo_real_ou_fora_criterio(bbox):
    frame_shape = (1080, 1920, 3)
    assert eh_capo_camera(bbox, frame_shape) is False


# ---------------------------------------------------------------------------
# 3. filtrar_caixas_redundantes
# ---------------------------------------------------------------------------

def test_filtrar_caixas_redundantes_vazio():
    assert filtrar_caixas_redundantes([]) == []


def test_filtrar_caixas_redundantes_remove_cabine_dentro_de_caminhao():
    caminhao = {"bbox": (100, 100, 500, 500), "cls": 7, "conf": 0.90}
    cabine = {"bbox": (150, 150, 300, 350), "cls": 7, "conf": 0.85}
    resultado = filtrar_caixas_redundantes([cabine, caminhao])
    assert resultado == [caminhao]


def test_filtrar_caixas_redundantes_mantem_caixas_distintas():
    carro_a = {"bbox": (10, 10, 110, 110), "cls": 2, "conf": 0.88}
    carro_b = {"bbox": (90, 10, 190, 110), "cls": 2, "conf": 0.86}  # inter = 20% <= 60%
    carro_c = {"bbox": (300, 200, 450, 350), "cls": 2, "conf": 0.92}
    resultado = filtrar_caixas_redundantes([carro_a, carro_b, carro_c])
    assert len(resultado) == 3


def test_filtrar_caixas_redundantes_limiar_customizado():
    maior = {"bbox": (0, 0, 200, 100), "cls": 2, "conf": 0.90}
    menor = {"bbox": (150, 0, 250, 100), "cls": 2, "conf": 0.80}  # inter/area_menor = 50%
    assert len(filtrar_caixas_redundantes([maior, menor], iomin_thresh=0.60)) == 2
    assert filtrar_caixas_redundantes([maior, menor], iomin_thresh=0.40) == [maior]


# ---------------------------------------------------------------------------
# 4. calcular_matriz_confiabilidade e helpers puros de VMMR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prob_1, prob_2, esperado",
    [
        # Alta: prob_1 >= 0.40 e margem >= 0.25
        (0.40, 0.15, "Alta"),
        (0.75, 0.20, "Alta"),
        (0.95, 0.00, "Alta"),
        # Média: margem >= 0.10, mas não atinge critérios de Alta
        (0.39, 0.10, "Média"),   # margem >= 0.25, porém prob_1 < 0.40
        (0.25, 0.15, "Média"),   # limite exato de margem == 0.10
        (0.55, 0.31, "Média"),   # margem == 0.24 < 0.25
        # Baixa / Ambígua: margem < 0.10
        (0.50, 0.41, "Baixa / Ambígua"),
        (0.20, 0.18, "Baixa / Ambígua"),
        (0.00, 0.00, "Baixa / Ambígua"),
    ],
)
def test_calcular_matriz_confiabilidade(prob_1, prob_2, esperado):
    assert calcular_matriz_confiabilidade(prob_1, prob_2) == esperado


@pytest.mark.parametrize(
    "modelo, esperado",
    [
        ("Volkswagen Fox", "Hatchback"),
        ("Toyota Corolla", "Sedan"),
        ("Jeep Compass", "SUV / Utilitário"),
        ("Fiat Strada", "Picape"),
        ("Chevrolet Spin", "Minivan"),
        ("Fiat Fiorino", "Furgão / Van"),
        ("Modelo Desconhecido", "Carro"),
        # Modelos separados por geração e nomes parciais
        ("Ford Fiesta Rocam", "Hatchback"),
        ("Ford New Fiesta", "Hatchback"),
        ("Ford Fiesta", "Hatchback"),
        ("Fiat Siena", "Sedan"),
        ("Fiat Grand Siena", "Sedan"),
        ("Fiat Uno Mille", "Hatchback"),
        ("Fiat Uno (Novo)", "Hatchback"),
        ("Fiat Uno", "Hatchback"),
        ("Chevrolet Prisma (G1)", "Sedan"),
        ("Chevrolet Prisma (G2)", "Sedan"),
        ("Chevrolet Prisma", "Sedan"),
        ("Ford Ka (Antigo)", "Hatchback"),
        ("Ford Ka", "Hatchback"),
        # Novos modelos e categorias do catálogo expandido
        ("Chevrolet Astra", "Hatchback"),
        ("Fiat Palio Weekend", "Perua / Station Wagon"),
        ("Volkswagen Kombi", "Furgão / Van"),
        ("Nissan March", "Hatchback"),
        ("BYD Song Plus", "SUV / Utilitário"),
        ("Peugeot 208", "Hatchback"),
        ("Chery Tiggo 8", "SUV / Utilitário"),
        ("Citroën C4 Lounge", "Sedan"),
        ("Citroën C4", "Hatchback"),
    ],
)
def test_inferir_categoria_por_modelo(modelo, esperado):
    assert inferir_categoria_por_modelo(modelo) == esperado


@pytest.mark.parametrize(
    "classe, esperado",
    [
        ("vw_fox", "Volkswagen Fox"),
        ("chevrolet-onix_2022", "Chevrolet Onix"),
        ("citroen_c3", "Citroën C3"),
        ("gm_s10", "Chevrolet S10"),
    ],
)
def test_formatar_nome_veiculo(classe, esperado):
    assert formatar_nome_veiculo(classe) == esperado


# ---------------------------------------------------------------------------
# 5. estimar_cor_lataria
# ---------------------------------------------------------------------------

def _criar_crop_bgr(bgr: tuple[int, int, int], altura: int = 120, largura: int = 160) -> np.ndarray:
    img = np.empty((altura, largura, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


@pytest.mark.parametrize(
    "bgr, esperado",
    [
        ((35, 35, 35), "Preto"),
        ((215, 215, 215), "Branco"),
        ((115, 115, 115), "Prata / Cinza"),
        ((30, 30, 205), "Vermelho"),
        ((205, 85, 20), "Azul"),
        ((35, 170, 35), "Verde"),
        ((20, 175, 225), "Amarelo / Laranja"),
        ((30, 60, 105), "Marrom"),
    ],
)
def test_estimar_cor_lataria_cores_representativas(bgr, esperado):
    crop = _criar_crop_bgr(bgr)
    assert estimar_cor_lataria(crop) == esperado


def test_estimar_cor_lataria_preto_com_reflexo_azulado_frio_moderado():
    # Simula lataria preta na sombra refletindo céu azul (B=50, G=41, R=22 -> H~100, S~143, V=50)
    crop = _criar_crop_bgr((50, 41, 22))
    assert estimar_cor_lataria(crop) == "Preto"


def test_estimar_cor_lataria_prata_com_reflexo_azulado_frio_moderado():
    # Simula lataria prata na sombra com viés frio (B=92, G=80, R=64 -> H~103, S~78, V=92)
    crop = _criar_crop_bgr((92, 80, 64))
    assert estimar_cor_lataria(crop) == "Prata / Cinza"


def test_estimar_cor_lataria_ignora_terco_superior_para_brisa():
    # Terço superior (vidro) com reflexo azul intenso, restante da lataria prata
    crop = _criar_crop_bgr((115, 115, 115), altura=120, largura=160)
    crop[:45, :] = (230, 110, 20)
    assert estimar_cor_lataria(crop) == "Prata / Cinza"


def test_estimar_cor_lataria_vermelho_wrap_around_hsv():
    # Mistura pixels próximos de H=2 e H=177 para garantir estatística circular no vermelho
    crop = _criar_crop_bgr((30, 35, 200), altura=120, largura=160)
    crop[:, :80] = (35, 30, 200)
    assert estimar_cor_lataria(crop) == "Vermelho"


def test_estimar_cor_lataria_entradas_invalidas():
    assert estimar_cor_lataria(None) == "Indefinida"
    assert estimar_cor_lataria(np.zeros((0, 0, 3), dtype=np.uint8)) == "Indefinida"
    assert estimar_cor_lataria(np.zeros((10, 10, 3), dtype=np.uint8)) == "Indefinida"
