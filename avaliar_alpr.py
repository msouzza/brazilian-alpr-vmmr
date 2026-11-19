"""
Script de avaliação quantitativa de QA para o pipeline Brazilian ALPR & VMMR.
Lê uma planilha .xlsx gerada por PlanilhaWriter contendo colunas manuais de gabarito
('Placa real', 'Modelo real', 'Condição' e opcionalmente 'Cor real') e calcula métricas
de acurácia por condição, Total Geral e calibração por Confiabilidade VMMR.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openpyxl import load_workbook

LIMIAR_AMOSTRA_MINIMA = 15
VALORES_SEM_PLACA = {"", "-", "SEM_PLACA", "SEM PLACA"}
VALORES_PLACA_ILEGIVEL = {"ILEGIVEL", "ILEGÍVEL"}
VALORES_FALSO_POSITIVO = {"SEM_VEICULO", "SEM VEICULO", "FALSO_POSITIVO", "FALSO POSITIVO"}


@dataclass
class RegistroAvaliacao:
    condicao: str
    placa_pred: str
    placa_real: str
    sem_placa_legivel: bool
    placa_exata: bool
    placa_ate_1_erro: bool
    modelo_real: str
    top1_modelo: str
    top3_modelos: list[str]
    modelo_top1_acerto: bool
    modelo_top3_acerto: bool
    confiabilidade_vmmr: str
    sem_placa: bool = False
    placa_ilegivel: bool = False
    cor_pred: str = ""
    cor_real: str = ""
    cor_correta: bool = False
    falso_positivo: bool = False


def normalizar_texto(texto: object) -> str:
    """Remove acentos, normaliza espaços e converte para minúsculas (casefold)."""
    if texto is None:
        return ""
    s = str(texto).strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s.casefold().strip()


def normalizar_placa(placa: object) -> str:
    """Normaliza placa para comparação alfanumérica em maiúsculas (preservando '?' e '$')."""
    if placa is None:
        return ""
    s = str(placa).strip().upper()
    if s in VALORES_SEM_PLACA:
        return s
    return re.sub(r"[^A-Z0-9?$]", "", s)


def extrair_nome_modelo(texto_descoberta: object) -> str:
    """
    Extrai o nome do modelo de strings como '1º Volkswagen Fox (73.5%)'.
    Retorna '' se vazio ou '-'.
    """
    if texto_descoberta is None:
        return ""
    s = str(texto_descoberta).strip()
    if not s or s == "-":
        return ""
    # Remove prefixo de ranking ('1º ', '2o ', '3° ', '1. ')
    s = re.sub(r"^\s*\d+\s*[º°o.]?\s*", "", s)
    # Remove sufixo de porcentagem (' (73.5%)', ' (100%)')
    s = re.sub(r"\s*\(\s*\d+(?:[.,]\d+)?\s*%\s*\)\s*$", "", s)
    return s.strip()


def eh_sem_placa(placa_real_bruta: object) -> bool:
    """Retorna True se o gabarito indicar veículo sem placa visível no enquadramento."""
    if placa_real_bruta is None:
        return True
    s = str(placa_real_bruta).strip().upper()
    return s in VALORES_SEM_PLACA


def eh_placa_ilegivel(placa_real_bruta: object) -> bool:
    """
    Retorna True se o gabarito indicar placa visível porém totalmente ilegível
    ('ILEGIVEL', 'ILEGÍVEL' ou composta apenas por '$', ex.: '$$$$$$$').
    """
    if placa_real_bruta is None:
        return False
    s = str(placa_real_bruta).strip().upper()
    if not s:
        return False
    if s in VALORES_PLACA_ILEGIVEL:
        return True
    limpo = re.sub(r"[\s-]", "", s)
    return bool(limpo) and all(ch == "$" for ch in limpo)


def eh_sem_placa_legivel(placa_real_bruta: object) -> bool:
    """Retorna True se o gabarito indicar veículo sem placa legível (SEM_PLACA ou ilegível)."""
    return eh_sem_placa(placa_real_bruta) or eh_placa_ilegivel(placa_real_bruta)


def eh_falso_positivo(placa_real_bruta: object, modelo_real_bruto: object = None) -> bool:
    """
    Retorna True se o gabarito indicar falso positivo (detecção sem veículo na foto),
    verificando tanto 'Placa real' quanto 'Modelo real'.
    """
    for valor in (placa_real_bruta, modelo_real_bruto):
        if valor is None:
            continue
        s = normalizar_texto(valor).upper()
        if s in VALORES_FALSO_POSITIVO:
            return True
    return False


def distancia_hamming_7(placa_a: str, placa_b: str) -> int | None:
    """
    Calcula distância de Hamming entre duas placas de 7 caracteres.
    Posições marcadas com '$' na segunda placa (gabarito incerto) não são penalizadas.
    """
    if len(placa_a) != 7 or len(placa_b) != 7:
        return None
    return sum(1 for a, b in zip(placa_a, placa_b) if b != "$" and a != b)


def formatar_taxa(acertos: int, total: int, min_amostras: int = LIMIAR_AMOSTRA_MINIMA) -> str:
    """
    Exibe percentual apenas quando total >= min_amostras;
    caso contrário exibe 'amostra pequena (n < 15)' sem percentual.
    """
    if total < min_amostras:
        return f"{acertos}/{total} — amostra pequena (n < {min_amostras})"
    pct = (acertos / total) * 100.0
    return f"{acertos}/{total} ({pct:.1f}%)"


def carregar_registros(
    caminho_xlsx: Path,
    aba: str | None = None,
    col_placa_real: str = "Placa real",
    col_modelo_real: str = "Modelo real",
    col_condicao: str = "Condição",
    col_cor_real: str = "Cor real",
) -> list[RegistroAvaliacao]:
    """Carrega e avalia cada linha da planilha .xlsx."""
    wb = load_workbook(caminho_xlsx, data_only=True)
    try:
        if aba:
            if aba not in wb.sheetnames:
                raise ValueError(f"Aba '{aba}' não encontrada na planilha. Abas disponíveis: {wb.sheetnames}")
            ws = wb[aba]
        elif "Placas" in wb.sheetnames:
            ws = wb["Placas"]
        else:
            ws = wb.active

        linhas = list(ws.iter_rows(values_only=True))
    finally:
        wb.close()

    if not linhas:
        return []

    cabecalho_raw = [str(c).strip() if c is not None else "" for c in linhas[0]]
    mapa_colunas: dict[str, int] = {}
    for idx, nome in enumerate(cabecalho_raw):
        if nome:
            mapa_colunas[normalizar_texto(nome)] = idx

    obrigatorias = [
        "Placa",
        "Descoberta 1 (%)",
        "Descoberta 2 (%)",
        "Descoberta 3 (%)",
        "Confiabilidade VMMR",
        col_placa_real,
        col_modelo_real,
        col_condicao,
    ]
    faltantes = [col for col in obrigatorias if normalizar_texto(col) not in mapa_colunas]
    if faltantes:
        raise ValueError(
            "Colunas obrigatórias ausentes na planilha: " + ", ".join(faltantes)
        )

    def _valor(linha: tuple, nome_col: str) -> object:
        idx = mapa_colunas.get(normalizar_texto(nome_col))
        if idx is None:
            return None
        return linha[idx] if idx < len(linha) else None

    tem_col_cor_real = normalizar_texto(col_cor_real) in mapa_colunas

    registros: list[RegistroAvaliacao] = []
    for linha in linhas[1:]:
        if not any(v is not None and str(v).strip() != "" for v in linha):
            continue

        cond_raw = _valor(linha, col_condicao)
        condicao = str(cond_raw).strip() if cond_raw is not None and str(cond_raw).strip() else "Não informada"

        placa_pred_raw = _valor(linha, "Placa")
        placa_real_raw = _valor(linha, col_placa_real)
        modelo_real_raw = _valor(linha, col_modelo_real)

        falso_positivo = eh_falso_positivo(placa_real_raw, modelo_real_raw)
        placa_pred = normalizar_placa(placa_pred_raw)

        d1 = extrair_nome_modelo(_valor(linha, "Descoberta 1 (%)"))
        d2 = extrair_nome_modelo(_valor(linha, "Descoberta 2 (%)"))
        d3 = extrair_nome_modelo(_valor(linha, "Descoberta 3 (%)"))

        top1_norm = normalizar_texto(d1)
        top3_norm = [m for m in (normalizar_texto(d1), normalizar_texto(d2), normalizar_texto(d3)) if m]

        cor_pred_norm = normalizar_texto(_valor(linha, "Cor"))
        if cor_pred_norm == "-":
            cor_pred_norm = ""

        if falso_positivo:
            sem_placa = False
            placa_ilegivel = False
            sem_placa_legivel = False
            placa_real = ""
            placa_exata = False
            placa_ate_1 = False
            modelo_real_norm = ""
            modelo_top1 = False
            modelo_top3 = False
            cor_real_norm = ""
            cor_correta = False
        else:
            sem_placa = eh_sem_placa(placa_real_raw)
            placa_ilegivel = eh_placa_ilegivel(placa_real_raw)
            sem_placa_legivel = sem_placa or placa_ilegivel

            placa_real = "" if sem_placa_legivel else normalizar_placa(placa_real_raw)

            if sem_placa_legivel:
                placa_exata = False
                placa_ate_1 = False
            else:
                dist = distancia_hamming_7(placa_pred, placa_real)
                if "$" in placa_real and len(placa_pred) == 7 and len(placa_real) == 7:
                    placa_exata = (dist == 0)
                else:
                    placa_exata = bool(placa_pred and placa_pred == placa_real)
                if placa_exata:
                    placa_ate_1 = True
                elif dist is not None and dist <= 1:
                    placa_ate_1 = True
                else:
                    placa_ate_1 = False

            modelo_real_norm = normalizar_texto(modelo_real_raw)
            if modelo_real_norm == "-":
                modelo_real_norm = ""

            modelo_top1 = bool(modelo_real_norm and top1_norm == modelo_real_norm)
            modelo_top3 = bool(modelo_real_norm and modelo_real_norm in top3_norm)

            if tem_col_cor_real:
                cor_real_norm = normalizar_texto(_valor(linha, col_cor_real))
                if cor_real_norm == "-":
                    cor_real_norm = ""
            else:
                cor_real_norm = ""

            cor_correta = bool(cor_real_norm and cor_pred_norm and cor_pred_norm == cor_real_norm)

        conf_raw = _valor(linha, "Confiabilidade VMMR")
        conf_vmmr = str(conf_raw).strip() if conf_raw is not None and str(conf_raw).strip() else "-"

        registros.append(
            RegistroAvaliacao(
                condicao=condicao,
                placa_pred=placa_pred,
                placa_real=placa_real,
                sem_placa_legivel=sem_placa_legivel,
                placa_exata=placa_exata,
                placa_ate_1_erro=placa_ate_1,
                modelo_real=modelo_real_norm,
                top1_modelo=top1_norm,
                top3_modelos=top3_norm,
                modelo_top1_acerto=modelo_top1,
                modelo_top3_acerto=modelo_top3,
                confiabilidade_vmmr=conf_vmmr,
                sem_placa=sem_placa,
                placa_ilegivel=placa_ilegivel,
                cor_pred=cor_pred_norm,
                cor_real=cor_real_norm,
                cor_correta=cor_correta,
                falso_positivo=falso_positivo,
            )
        )

    return registros


def _agregar_grupo(itens: Iterable[RegistroAvaliacao]) -> dict[str, int]:
    lista = list(itens)
    total = len(lista)
    falso_positivo = sum(1 for r in lista if r.falso_positivo)
    sem_placa = sum(
        1
        for r in lista
        if not r.falso_positivo and (r.sem_placa or (r.sem_placa_legivel and not r.placa_ilegivel))
    )
    ilegivel = sum(1 for r in lista if not r.falso_positivo and r.placa_ilegivel)
    sem_placa_legivel = sum(1 for r in lista if not r.falso_positivo and r.sem_placa_legivel)
    placas_validas = total - falso_positivo - sem_placa_legivel
    placa_exata = sum(
        1 for r in lista if not r.falso_positivo and not r.sem_placa_legivel and r.placa_exata
    )
    placa_ate_1 = sum(
        1 for r in lista if not r.falso_positivo and not r.sem_placa_legivel and r.placa_ate_1_erro
    )

    com_modelo = [r for r in lista if not r.falso_positivo and r.modelo_real]
    modelos_validos = len(com_modelo)
    modelo_top1 = sum(1 for r in com_modelo if r.modelo_top1_acerto)
    modelo_top3 = sum(1 for r in com_modelo if r.modelo_top3_acerto)

    com_cor = [r for r in lista if not r.falso_positivo and r.cor_real]
    cores_validas = len(com_cor)
    cor_correta = sum(1 for r in com_cor if r.cor_correta)

    return {
        "total": total,
        "falso_positivo": falso_positivo,
        "sem_placa": sem_placa,
        "ilegivel": ilegivel,
        "sem_placa_legivel": sem_placa_legivel,
        "placas_validas": placas_validas,
        "placa_exata": placa_exata,
        "placa_ate_1": placa_ate_1,
        "modelos_validos": modelos_validos,
        "modelo_top1": modelo_top1,
        "modelo_top3": modelo_top3,
        "cores_validas": cores_validas,
        "cor_correta": cor_correta,
    }


def gerar_relatorio_markdown(
    registros: list[RegistroAvaliacao],
    min_amostras: int = LIMIAR_AMOSTRA_MINIMA,
) -> str:
    """Gera o relatório em formato Markdown com métricas por Condição e calibração VMMR."""
    if not registros:
        return "# Relatório de Avaliação ALPR & VMMR\n\nNenhum registro encontrado na planilha.\n"

    por_condicao: dict[str, list[RegistroAvaliacao]] = defaultdict(list)
    for r in registros:
        por_condicao[r.condicao].append(r)

    linhas_md = [
        "# Relatório de Avaliação ALPR & VMMR",
        "",
        f"> Regra estatística de QA: percentuais exibidos somente quando `n >= {min_amostras}` amostras válidas.",
        "",
        "## Métricas por Condição",
        "",
        "| Condição | Conferidos | Sem placa | Ilegível | Falso positivo | Placa exata | Placa (<= 1 erro) | Modelo Top 1 | Modelo Top 3 | Cor correta |",
        "| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- | :--- |",
    ]

    for condicao in sorted(por_condicao.keys()):
        m = _agregar_grupo(por_condicao[condicao])
        linhas_md.append(
            f"| {condicao} | {m['total']} | {m['sem_placa']} | {m['ilegivel']} | {m['falso_positivo']} | "
            f"{formatar_taxa(m['placa_exata'], m['placas_validas'], min_amostras)} | "
            f"{formatar_taxa(m['placa_ate_1'], m['placas_validas'], min_amostras)} | "
            f"{formatar_taxa(m['modelo_top1'], m['modelos_validos'], min_amostras)} | "
            f"{formatar_taxa(m['modelo_top3'], m['modelos_validos'], min_amostras)} | "
            f"{formatar_taxa(m['cor_correta'], m['cores_validas'], min_amostras)} |"
        )

    total_m = _agregar_grupo(registros)
    linhas_md.append(
        f"| **Total Geral** | **{total_m['total']}** | **{total_m['sem_placa']}** | **{total_m['ilegivel']}** | **{total_m['falso_positivo']}** | "
        f"**{formatar_taxa(total_m['placa_exata'], total_m['placas_validas'], min_amostras)}** | "
        f"**{formatar_taxa(total_m['placa_ate_1'], total_m['placas_validas'], min_amostras)}** | "
        f"**{formatar_taxa(total_m['modelo_top1'], total_m['modelos_validos'], min_amostras)}** | "
        f"**{formatar_taxa(total_m['modelo_top3'], total_m['modelos_validos'], min_amostras)}** | "
        f"**{formatar_taxa(total_m['cor_correta'], total_m['cores_validas'], min_amostras)}** |"
    )

    # Tabela de calibração por Confiabilidade VMMR
    por_conf: dict[str, list[RegistroAvaliacao]] = defaultdict(list)
    for r in registros:
        if not r.falso_positivo:
            por_conf[r.confiabilidade_vmmr].append(r)

    ordem_padrao = ["Alta", "Média", "Baixa / Ambígua"]
    chaves_extras = sorted(k for k in por_conf.keys() if k not in ordem_padrao)
    chaves_ordenadas = [k for k in ordem_padrao if k in por_conf] + chaves_extras

    linhas_md.extend(
        [
            "",
            "## Calibração por Confiabilidade VMMR",
            "",
            "| Confiabilidade VMMR | Amostras com modelo real | Modelo Top 1 | Modelo Top 3 |",
            "| :--- | :---: | :--- | :--- |",
        ]
    )

    for nivel in chaves_ordenadas:
        m = _agregar_grupo(por_conf[nivel])
        linhas_md.append(
            f"| {nivel} | {m['modelos_validos']} | "
            f"{formatar_taxa(m['modelo_top1'], m['modelos_validos'], min_amostras)} | "
            f"{formatar_taxa(m['modelo_top3'], m['modelos_validos'], min_amostras)} |"
        )

    linhas_md.append("")
    return "\n".join(linhas_md)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Avalia métricas reais de ALPR e VMMR a partir da planilha conferida."
    )
    parser.add_argument(
        "planilha",
        type=Path,
        help="Caminho para o arquivo .xlsx (ex: data/placas_detectadas.xlsx)",
    )
    parser.add_argument(
        "--saida",
        "-o",
        type=Path,
        default=None,
        help="Caminho opcional para salvar o relatório em Markdown (ex: resultados.md)",
    )
    parser.add_argument(
        "--aba",
        type=str,
        default=None,
        help="Nome da aba da planilha (padrão: 'Placas' se existir, senão a aba ativa)",
    )
    parser.add_argument(
        "--col-placa-real",
        type=str,
        default="Placa real",
        help="Nome da coluna com o gabarito de placa (padrão: 'Placa real')",
    )
    parser.add_argument(
        "--col-modelo-real",
        type=str,
        default="Modelo real",
        help="Nome da coluna com o gabarito de modelo (padrão: 'Modelo real')",
    )
    parser.add_argument(
        "--col-condicao",
        type=str,
        default="Condição",
        help="Nome da coluna com a condição de captura (padrão: 'Condição')",
    )
    parser.add_argument(
        "--col-cor-real",
        type=str,
        default="Cor real",
        help="Nome da coluna com o gabarito de cor (padrão: 'Cor real')",
    )

    args = parser.parse_args(argv)
    if not args.planilha.exists():
        print(f"Erro: arquivo não encontrado: {args.planilha}", file=sys.stderr)
        return 1

    registros = carregar_registros(
        caminho_xlsx=args.planilha,
        aba=args.aba,
        col_placa_real=args.col_placa_real,
        col_modelo_real=args.col_modelo_real,
        col_condicao=args.col_condicao,
        col_cor_real=args.col_cor_real,
    )
    relatorio = gerar_relatorio_markdown(registros)
    print(relatorio)

    if args.saida is not None:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(relatorio, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
