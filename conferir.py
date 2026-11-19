"""
Painel web local de conferência rápida para validação manual (QA)
da planilha gerada pelo pipeline Brazilian ALPR & VMMR.

Uso:
    python conferir.py [caminho_xlsx] [--porta 8765] [--no-browser]
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import sys
import threading
import webbrowser
from datetime import date, datetime, time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from openpyxl import load_workbook

import avaliar_alpr

COLUNAS_GABARITO = ("Placa real", "Modelo real", "Cor real", "Condição")

CORES_PADRAO = (
    "Prata / Cinza",
    "Branco",
    "Preto",
    "Azul",
    "Vermelho",
    "Verde",
    "Amarelo / Laranja",
    "Marrom",
    "Roxo / Vinho",
    "Outra",
)

CONDICOES_PADRAO = (
    "dia_traseira",
    "dia_frontal",
    "noite_traseira",
    "noite_frontal",
    "moto",
    "lateral",
)

_PLANILHA_LOCK = threading.RLock()


def resolver_caminho_gabarito(
    caminho_xlsx: Path, caminho_gabarito: Path | None = None
) -> Path:
    """Retorna o caminho do arquivo JSON de backup do gabarito."""
    if caminho_gabarito is not None:
        return Path(caminho_gabarito)
    return Path(caminho_xlsx).parent / "gabarito_qa.json"


def _extrair_nome_foto(foto: Any) -> str:
    """Extrai o nome do arquivo da foto se o caminho informado for válido."""
    bruto = str(foto or "").strip()
    if not bruto or bruto == "-":
        return ""
    nome = Path(bruto).name.strip()
    if not nome or nome in {"-", "."}:
        return ""
    return nome


def chave_gabarito_registro(
    foto: Any = "", video: Any = "", data: Any = "", hora: Any = ""
) -> str:
    """
    Gera a chave de identificação do veículo para persistência em gabarito_qa.json.
    Usa o nome do arquivo da foto como chave primária e 'video|data|hora' como
    chave secundária caso a foto não exista.
    """
    foto_nome = _extrair_nome_foto(foto)
    if foto_nome:
        return foto_nome
    video_s = str(video or "").strip()
    data_s = str(data or "").strip()
    hora_s = str(hora or "").strip()
    if video_s or data_s or hora_s:
        return f"{video_s}|{data_s}|{hora_s}"
    return ""


def carregar_gabarito_json(caminho_gabarito: Path) -> dict[str, dict[str, str]]:
    """Carrega o dicionário de backup de gabarito a partir do JSON, se existir."""
    caminho_gabarito = Path(caminho_gabarito)
    if not caminho_gabarito.exists():
        return {}
    try:
        conteudo = caminho_gabarito.read_text(encoding="utf-8").strip()
        if not conteudo:
            return {}
        dados = json.loads(conteudo)
        if isinstance(dados, dict):
            return dados
    except Exception:
        return {}
    return {}


def salvar_gabarito_json(
    caminho_gabarito: Path, gabarito: dict[str, dict[str, str]]
) -> None:
    """Persiste o dicionário de backup de gabarito em JSON UTF-8."""
    caminho_gabarito = Path(caminho_gabarito)
    caminho_gabarito.parent.mkdir(parents=True, exist_ok=True)
    caminho_gabarito.write_text(
        json.dumps(gabarito, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def carregar_catalogo_modelos() -> list[str]:
    """Importa opcionalmente CATALOGO_FROTA_BR para sugestões de modelos."""
    try:
        from src.vehicle_classifier import CATALOGO_FROTA_BR

        return sorted(str(k) for k in CATALOGO_FROTA_BR.keys())
    except Exception:
        return []


def _selecionar_aba(wb, aba: str | None = None):
    if aba:
        if aba not in wb.sheetnames:
            raise ValueError(
                f"Aba '{aba}' não encontrada na planilha. Abas disponíveis: {wb.sheetnames}"
            )
        return wb[aba]
    if "Placas" in wb.sheetnames:
        return wb["Placas"]
    return wb.active


def _mapear_cabecalho(ws) -> tuple[dict[str, int], int]:
    """
    Mapeia nomes normalizados de coluna para índice 1-based no openpyxl
    e retorna também a última coluna não vazia do cabeçalho.
    """
    mapa: dict[str, int] = {}
    ultima_col = 0
    if ws.max_row < 1:
        return mapa, ultima_col

    for col_idx, cell in enumerate(ws[1], start=1):
        val = cell.value
        if val is not None and str(val).strip() != "":
            ultima_col = col_idx
            chave = avaliar_alpr.normalizar_texto(val)
            if chave and chave not in mapa:
                mapa[chave] = col_idx
    return mapa, ultima_col


def _formatar_celula(valor: Any) -> str:
    if valor is None:
        return ""
    if isinstance(valor, datetime):
        return valor.strftime("%d/%m/%Y %H:%M:%S")
    if isinstance(valor, date):
        return valor.strftime("%d/%m/%Y")
    if isinstance(valor, time):
        return valor.strftime("%H:%M:%S")
    return str(valor).strip()


def _formatar_conf_ocr(valor: Any) -> str | float:
    if valor is None or str(valor).strip() == "":
        return ""
    if isinstance(valor, (int, float)):
        return round(float(valor), 4)
    return str(valor).strip()


def _garantir_colunas_gabarito_ws(ws) -> tuple[dict[str, int], dict[str, int], bool]:
    """
    Garante as colunas de gabarito na worksheet `ws` sem alterar colunas existentes.
    Retorna (mapa_cabecalho_normalizado, indices_gabarito, alterado).
    """
    mapa, ultima_col = _mapear_cabecalho(ws)
    alterado = False
    indices: dict[str, int] = {}
    for col_nome in COLUNAS_GABARITO:
        chave = avaliar_alpr.normalizar_texto(col_nome)
        if chave in mapa:
            indices[col_nome] = mapa[chave]
        else:
            ultima_col += 1
            ws.cell(row=1, column=ultima_col, value=col_nome)
            mapa[chave] = ultima_col
            indices[col_nome] = ultima_col
            alterado = True
    return mapa, indices, alterado


def garantir_colunas_gabarito(caminho_xlsx: Path, aba: str | None = None) -> dict[str, int]:
    """
    Garante a existência das colunas 'Placa real', 'Modelo real', 'Cor real' e 'Condição'
    ao final do cabeçalho sem alterar ou sobrescrever colunas existentes.
    Retorna o mapeamento {nome_coluna_gabarito: indice_1_based}.
    """
    caminho_xlsx = Path(caminho_xlsx)
    with _PLANILHA_LOCK:
        wb = load_workbook(caminho_xlsx)
        try:
            ws = _selecionar_aba(wb, aba=aba)
            _, indices, alterado = _garantir_colunas_gabarito_ws(ws)
            if alterado:
                wb.save(caminho_xlsx)
            return indices
        finally:
            wb.close()


def _ler_celula_mapeada(ws, row_idx: int, mapa: dict[str, int], nome_col: str) -> str:
    col_idx = mapa.get(avaliar_alpr.normalizar_texto(nome_col))
    if col_idx is None:
        return ""
    return _formatar_celula(ws.cell(row=row_idx, column=col_idx).value)


def restaurar_gabarito_na_planilha(
    caminho_xlsx: Path,
    aba: str | None = None,
    caminho_gabarito: Path | None = None,
) -> int:
    """
    Restaura automaticamente os campos ('Placa real', 'Modelo real', 'Cor real', 'Condição')
    em linhas vazias do .xlsx a partir de gabarito_qa.json.
    Retorna o total de linhas restauradas e salvas na planilha.
    """
    caminho_xlsx = Path(caminho_xlsx)
    dest_gabarito = resolver_caminho_gabarito(caminho_xlsx, caminho_gabarito)
    gabarito = carregar_gabarito_json(dest_gabarito)
    if not gabarito:
        return 0

    with _PLANILHA_LOCK:
        wb = load_workbook(caminho_xlsx)
        try:
            ws = _selecionar_aba(wb, aba=aba)
            if ws.max_row < 2:
                return 0

            mapa, _ = _mapear_cabecalho(ws)
            indices_gabarito: dict[str, int] | None = None
            restaurados = 0

            for row_idx in range(2, ws.max_row + 1):
                valores_linha = [
                    ws.cell(row=row_idx, column=c).value
                    for c in range(1, ws.max_column + 1)
                ]
                if not any(v is not None and str(v).strip() != "" for v in valores_linha):
                    continue

                placa_real = _ler_celula_mapeada(ws, row_idx, mapa, "Placa real").upper()
                modelo_real = _ler_celula_mapeada(ws, row_idx, mapa, "Modelo real")
                cor_real = _ler_celula_mapeada(ws, row_idx, mapa, "Cor real")
                condicao = _ler_celula_mapeada(ws, row_idx, mapa, "Condição")
                if placa_real or modelo_real or cor_real or condicao:
                    continue

                foto = _ler_celula_mapeada(ws, row_idx, mapa, "Foto")
                video = _ler_celula_mapeada(ws, row_idx, mapa, "Vídeo de origem")
                data = _ler_celula_mapeada(ws, row_idx, mapa, "Data")
                hora = _ler_celula_mapeada(ws, row_idx, mapa, "Hora")

                chave = chave_gabarito_registro(
                    foto=foto, video=video, data=data, hora=hora
                )
                if not chave:
                    continue

                entrada = gabarito.get(chave)
                if not isinstance(entrada, dict):
                    continue

                p_rest = str(entrada.get("placa_real") or "").strip().upper()
                m_rest = str(entrada.get("modelo_real") or "").strip()
                cor_rest = str(entrada.get("cor_real") or "").strip()
                c_rest = str(entrada.get("condicao") or "").strip()
                if not (p_rest or m_rest or cor_rest or c_rest):
                    continue

                if indices_gabarito is None:
                    mapa, indices_gabarito, _ = _garantir_colunas_gabarito_ws(ws)

                ws.cell(row=row_idx, column=indices_gabarito["Placa real"], value=p_rest or None)
                ws.cell(row=row_idx, column=indices_gabarito["Modelo real"], value=m_rest or None)
                ws.cell(row=row_idx, column=indices_gabarito["Cor real"], value=cor_rest or None)
                ws.cell(row=row_idx, column=indices_gabarito["Condição"], value=c_rest or None)
                restaurados += 1

            if restaurados > 0:
                try:
                    wb.save(caminho_xlsx)
                except PermissionError:
                    return 0

            return restaurados
        finally:
            wb.close()


def sincronizar_gabarito_da_planilha(
    caminho_xlsx: Path,
    aba: str | None = None,
    caminho_gabarito: Path | None = None,
) -> int:
    """
    Exporta para gabarito_qa.json todas as linhas já conferidas no .xlsx.
    Preserva entradas anteriores de outros vídeos/lotes que não estejam nesta planilha.
    Retorna a quantidade de registros conferidos sincronizados.
    """
    caminho_xlsx = Path(caminho_xlsx)
    dest_gabarito = resolver_caminho_gabarito(caminho_xlsx, caminho_gabarito)
    registros = carregar_registros_planilha(
        caminho_xlsx, aba=aba, caminho_gabarito=dest_gabarito
    )
    with _PLANILHA_LOCK:
        gabarito = carregar_gabarito_json(dest_gabarito)
        sincronizados = 0
        for r in registros:
            if not r["conferido"]:
                continue
            chave = chave_gabarito_registro(
                foto=r["foto"], video=r["video"], data=r["data"], hora=r["hora"]
            )
            if not chave:
                continue
            gabarito[chave] = {
                "placa_real": r["placa_real"],
                "modelo_real": r["modelo_real"],
                "cor_real": r["cor_real"],
                "condicao": r["condicao"],
                "video": r["video"],
                "data": r["data"],
                "hora": r["hora"],
                "foto_nome": _extrair_nome_foto(r["foto"]),
            }
            sincronizados += 1
        salvar_gabarito_json(dest_gabarito, gabarito)
        return sincronizados


def carregar_registros_planilha(
    caminho_xlsx: Path,
    aba: str | None = None,
    caminho_gabarito: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Lê os registros da planilha em modo somente leitura, preservando o índice
    real da linha no Excel (1-based, começando na linha 2).
    Restaura automaticamente linhas vazias com correspondência em gabarito_qa.json.
    """
    caminho_xlsx = Path(caminho_xlsx)
    dest_gabarito = resolver_caminho_gabarito(caminho_xlsx, caminho_gabarito)
    restaurar_gabarito_na_planilha(
        caminho_xlsx, aba=aba, caminho_gabarito=dest_gabarito
    )
    gabarito = carregar_gabarito_json(dest_gabarito)

    with _PLANILHA_LOCK:
        wb = load_workbook(caminho_xlsx, data_only=True, read_only=True)
        try:
            ws = _selecionar_aba(wb, aba=aba)
            iter_linhas = ws.iter_rows(values_only=True)
            try:
                cabecalho = next(iter_linhas)
            except StopIteration:
                return []

            mapa: dict[str, int] = {}
            for idx, nome in enumerate(cabecalho):
                if nome is not None and str(nome).strip() != "":
                    chave = avaliar_alpr.normalizar_texto(nome)
                    if chave and chave not in mapa:
                        mapa[chave] = idx

            def _get(linha: tuple[Any, ...], nome_col: str) -> Any:
                idx = mapa.get(avaliar_alpr.normalizar_texto(nome_col))
                if idx is None or idx >= len(linha):
                    return None
                return linha[idx]

            registros: list[dict[str, Any]] = []
            for row_idx, linha in enumerate(iter_linhas, start=2):
                if not any(v is not None and str(v).strip() != "" for v in linha):
                    continue

                desc_1 = _formatar_celula(_get(linha, "Descoberta 1 (%)"))
                desc_2 = _formatar_celula(_get(linha, "Descoberta 2 (%)"))
                desc_3 = _formatar_celula(_get(linha, "Descoberta 3 (%)"))

                data_str = _formatar_celula(_get(linha, "Data"))
                hora_str = _formatar_celula(_get(linha, "Hora"))
                foto_str = _formatar_celula(_get(linha, "Foto"))
                video_str = _formatar_celula(_get(linha, "Vídeo de origem"))

                placa_real = _formatar_celula(_get(linha, "Placa real")).upper()
                modelo_real = _formatar_celula(_get(linha, "Modelo real"))
                cor_real = _formatar_celula(_get(linha, "Cor real"))
                condicao = _formatar_celula(_get(linha, "Condição"))

                if not (placa_real or modelo_real or cor_real or condicao) and gabarito:
                    chave = chave_gabarito_registro(
                        foto=foto_str, video=video_str, data=data_str, hora=hora_str
                    )
                    entrada = gabarito.get(chave) if chave else None
                    if isinstance(entrada, dict):
                        placa_real = str(entrada.get("placa_real") or "").strip().upper()
                        modelo_real = str(entrada.get("modelo_real") or "").strip()
                        cor_real = str(entrada.get("cor_real") or "").strip()
                        condicao = str(entrada.get("condicao") or "").strip()

                registros.append(
                    {
                        "row": row_idx,
                        "placa": _formatar_celula(_get(linha, "Placa")),
                        "tipo": _formatar_celula(_get(linha, "Tipo")),
                        "cor": _formatar_celula(_get(linha, "Cor")),
                        "marca_modelo": _formatar_celula(_get(linha, "Marca / Modelo")),
                        "desc_1": desc_1,
                        "desc_2": desc_2,
                        "desc_3": desc_3,
                        "desc_1_modelo": avaliar_alpr.extrair_nome_modelo(desc_1),
                        "desc_2_modelo": avaliar_alpr.extrair_nome_modelo(desc_2),
                        "desc_3_modelo": avaliar_alpr.extrair_nome_modelo(desc_3),
                        "confiabilidade": _formatar_celula(_get(linha, "Confiabilidade VMMR")),
                        "data": data_str,
                        "hora": hora_str,
                        "conf_ocr": _formatar_conf_ocr(_get(linha, "Confiança OCR")),
                        "foto": foto_str,
                        "video": video_str,
                        "placa_real": placa_real,
                        "modelo_real": modelo_real,
                        "cor_real": cor_real,
                        "condicao": condicao,
                        "conferido": bool(placa_real or modelo_real or cor_real or condicao),
                    }
                )
            return registros
        finally:
            wb.close()


def salvar_registro(
    caminho_xlsx: Path,
    row: int,
    placa_real: str = "",
    modelo_real: str = "",
    condicao: str = "",
    cor_real: str = "",
    aba: str | None = None,
    caminho_gabarito: Path | None = None,
) -> dict[str, Any]:
    """
    Grava placa_real, modelo_real, cor_real e condicao diretamente na linha `row`
    (1-based) da planilha, garantindo as colunas de gabarito se ausentes,
    e sincroniza o backup persistente em gabarito_qa.json.
    """
    if not isinstance(row, int) or row < 2:
        raise ValueError(f"Índice de linha inválido: {row!r}. Deve ser >= 2.")

    caminho_xlsx = Path(caminho_xlsx)
    dest_gabarito = resolver_caminho_gabarito(caminho_xlsx, caminho_gabarito)
    placa_limpa = str(placa_real or "").strip().upper()
    modelo_limpo = str(modelo_real or "").strip()
    cor_limpa = str(cor_real or "").strip()
    condicao_limpa = str(condicao or "").strip()

    with _PLANILHA_LOCK:
        wb = load_workbook(caminho_xlsx)
        try:
            ws = _selecionar_aba(wb, aba=aba)
            if row > ws.max_row:
                raise ValueError(
                    f"Linha {row} fora dos limites da planilha (máximo: {ws.max_row})."
                )

            mapa, indices, _ = _garantir_colunas_gabarito_ws(ws)

            ws.cell(row=row, column=indices["Placa real"], value=placa_limpa or None)
            ws.cell(row=row, column=indices["Modelo real"], value=modelo_limpo or None)
            ws.cell(row=row, column=indices["Cor real"], value=cor_limpa or None)
            ws.cell(row=row, column=indices["Condição"], value=condicao_limpa or None)

            foto = _ler_celula_mapeada(ws, row, mapa, "Foto")
            video = _ler_celula_mapeada(ws, row, mapa, "Vídeo de origem")
            data = _ler_celula_mapeada(ws, row, mapa, "Data")
            hora = _ler_celula_mapeada(ws, row, mapa, "Hora")

            wb.save(caminho_xlsx)
        finally:
            wb.close()

        chave = chave_gabarito_registro(foto=foto, video=video, data=data, hora=hora)
        if chave:
            gabarito = carregar_gabarito_json(dest_gabarito)
            if placa_limpa or modelo_limpo or cor_limpa or condicao_limpa:
                gabarito[chave] = {
                    "placa_real": placa_limpa,
                    "modelo_real": modelo_limpo,
                    "cor_real": cor_limpa,
                    "condicao": condicao_limpa,
                    "video": video,
                    "data": data,
                    "hora": hora,
                    "foto_nome": _extrair_nome_foto(foto),
                }
            else:
                gabarito.pop(chave, None)
            salvar_gabarito_json(dest_gabarito, gabarito)

    return {
        "row": row,
        "placa_real": placa_limpa,
        "modelo_real": modelo_limpo,
        "cor_real": cor_limpa,
        "condicao": condicao_limpa,
        "conferido": bool(placa_limpa or modelo_limpo or cor_limpa or condicao_limpa),
    }


def resolver_caminho_foto(
    caminho_xlsx: Path, row: int, aba: str | None = None
) -> tuple[Path | None, str]:
    """
    Busca o caminho salvo na coluna 'Foto' da linha `row` e resolve
    para um arquivo existente no disco. Retorna (Path | None, caminho_original).
    """
    if row < 2:
        return None, ""

    caminho_xlsx = Path(caminho_xlsx)
    with _PLANILHA_LOCK:
        wb = load_workbook(caminho_xlsx, data_only=True, read_only=True)
        try:
            ws = _selecionar_aba(wb, aba=aba)
            iter_linhas = ws.iter_rows(values_only=True)
            try:
                cabecalho = next(iter_linhas)
            except StopIteration:
                return None, ""

            idx_foto = None
            for idx, nome in enumerate(cabecalho):
                if nome is not None and avaliar_alpr.normalizar_texto(nome) == "foto":
                    idx_foto = idx
                    break

            if idx_foto is None:
                return None, ""

            foto_raw = ""
            for cur_row, linha in enumerate(iter_linhas, start=2):
                if cur_row == row:
                    if idx_foto < len(linha):
                        foto_raw = _formatar_celula(linha[idx_foto])
                    break
        finally:
            wb.close()

    if not foto_raw or foto_raw == "-":
        return None, foto_raw

    candidato = Path(foto_raw)
    tentativas = [candidato]
    if not candidato.is_absolute():
        tentativas.extend(
            [
                caminho_xlsx.resolve().parent / candidato,
                caminho_xlsx.resolve().parent.parent / candidato,
                Path.cwd() / candidato,
            ]
        )

    for p in tentativas:
        if p.exists() and p.is_file():
            return p, foto_raw

    return None, foto_raw


def gerar_relatorio_qa(
    caminho_xlsx: Path,
    aba: str | None = None,
    caminho_gabarito: Path | None = None,
) -> str:
    """Executa a lógica de avaliar_alpr.py e retorna o relatório Markdown."""
    garantir_colunas_gabarito(caminho_xlsx, aba=aba)
    restaurar_gabarito_na_planilha(
        caminho_xlsx, aba=aba, caminho_gabarito=caminho_gabarito
    )
    registros = avaliar_alpr.carregar_registros(caminho_xlsx, aba=aba)
    return avaliar_alpr.gerar_relatorio_markdown(registros)


def _construir_html(caminho_xlsx: Path, modelos_catalogo: list[str]) -> str:
    opcoes_datalist = "\n".join(
        f'      <option value="{html.escape(m)}"></option>' for m in modelos_catalogo
    )
    botoes_cor = "\n".join(
        f'          <button type="button" class="chip-btn" data-cor="{html.escape(c)}">{html.escape(c)}</button>'
        for c in CORES_PADRAO
    )
    botoes_condicao = "\n".join(
        f'          <button type="button" class="chip-btn" data-cond="{html.escape(c)}">{html.escape(c)}</button>'
        for c in CONDICOES_PADRAO
    )
    nome_arquivo = html.escape(str(caminho_xlsx))

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Conferência Rápida QA — ALPR & VMMR</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #1e293b;
      --panel-2: #0b1220;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
      --success: #22c55e;
      --success-bg: rgba(34, 197, 94, 0.16);
      --warn: #f59e0b;
      --danger: #ef4444;
      --danger-bg: rgba(239, 68, 68, 0.16);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      flex-wrap: wrap;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }}
    .brand h1 {{
      margin: 0;
      font-size: 16px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .file-pill {{
      font-size: 12px;
      color: var(--muted);
      background: var(--panel-2);
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 999px;
      max-width: 340px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .progress-wrap {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      font-weight: 600;
    }}
    .progress-bar {{
      width: 140px;
      height: 8px;
      background: var(--panel-2);
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid var(--border);
    }}
    .progress-fill {{
      height: 100%;
      background: var(--success);
      width: 0%;
      transition: width 0.2s ease;
    }}
    .segmented {{
      display: inline-flex;
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 2px;
    }}
    .segmented button {{
      background: transparent;
      color: var(--muted);
      border: none;
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
    }}
    .segmented button.active {{
      background: var(--accent);
      color: #082f49;
    }}
    .btn {{
      border: 1px solid var(--border);
      background: #273449;
      color: var(--text);
      padding: 7px 13px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.12s ease, border-color 0.12s ease;
    }}
    .btn:hover:not(:disabled) {{
      background: #334155;
      border-color: #475569;
    }}
    .btn:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    .btn-primary {{
      background: var(--accent);
      color: #082f49;
      border-color: var(--accent);
    }}
    .btn-primary:hover:not(:disabled) {{
      background: var(--accent-hover);
      border-color: var(--accent-hover);
    }}
    .btn-success {{
      background: var(--success);
      color: #052e16;
      border-color: var(--success);
    }}
    .btn-success:hover:not(:disabled) {{
      background: #16a34a;
      border-color: #16a34a;
    }}
    .btn-sm {{
      padding: 5px 10px;
      font-size: 12px;
    }}
    main {{
      flex: 1;
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(380px, 1fr);
      min-height: 0;
    }}
    .photo-pane {{
      position: relative;
      background: #020617;
      display: flex;
      flex-direction: column;
      min-height: 0;
      border-right: 1px solid var(--border);
    }}
    .photo-topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 14px;
      background: rgba(15, 23, 42, 0.88);
      border-bottom: 1px solid var(--border);
      font-size: 12px;
      color: var(--muted);
      gap: 10px;
    }}
    .photo-stage {{
      flex: 1;
      position: relative;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 10px;
      user-select: none;
    }}
    .photo-stage img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: 6px;
      cursor: zoom-in;
      transition: transform 0.14s ease-out;
      transform-origin: center center;
    }}
    .photo-stage.zoomed img {{
      cursor: zoom-out;
      transform: scale(2.6);
      transition: transform 0.04s linear;
    }}
    .photo-error {{
      display: none;
      text-align: center;
      padding: 24px;
      max-width: 520px;
      background: var(--panel);
      border: 1px dashed var(--danger);
      border-radius: 12px;
      color: #fecaca;
      font-size: 13px;
      line-height: 1.5;
    }}
    .form-pane {{
      background: var(--panel);
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .form-scroll {{
      flex: 1;
      overflow-y: auto;
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .meta-card {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .meta-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .nav-index {{
      font-size: 13px;
      font-weight: 700;
      color: var(--accent);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      border: 1px solid var(--border);
      background: var(--panel);
    }}
    .badge-ok {{
      color: #86efac;
      background: var(--success-bg);
      border-color: rgba(34, 197, 94, 0.4);
    }}
    .badge-pending {{
      color: #fde68a;
      background: rgba(245, 158, 11, 0.14);
      border-color: rgba(245, 158, 11, 0.4);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 12px;
      font-size: 12px;
    }}
    .meta-item span {{
      color: var(--muted);
    }}
    .meta-item strong {{
      color: var(--text);
      font-weight: 600;
    }}
    .plate-big {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 16px;
      letter-spacing: 1px;
      color: #7dd3fc;
    }}
    .field-card {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 9px;
    }}
    .field-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }}
    .field-header label {{
      font-size: 13px;
      font-weight: 700;
    }}
    .field-hint {{
      font-size: 11px;
      color: var(--muted);
    }}
    input[type="text"] {{
      width: 100%;
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px 12px;
      font-size: 15px;
      outline: none;
    }}
    input[type="text"]:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }}
    .input-plate {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-weight: 700;
      letter-spacing: 1.5px;
      text-transform: uppercase;
    }}
    .btn-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip-btn {{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      padding: 6px 11px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.12s ease;
    }}
    .chip-btn:hover:not(:disabled) {{
      border-color: var(--accent);
      color: #bae6fd;
    }}
    .chip-btn.active {{
      background: var(--accent);
      color: #082f49;
      border-color: var(--accent);
    }}
    .chip-btn:disabled {{
      opacity: 0.4;
      cursor: not-allowed;
    }}
    .fp-banner {{
      background: var(--danger-bg);
      border: 1px solid rgba(239, 68, 68, 0.45);
      border-radius: 10px;
      padding: 11px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      cursor: pointer;
      transition: background 0.12s ease, border-color 0.12s ease;
    }}
    .fp-banner:hover {{
      border-color: var(--danger);
    }}
    .fp-banner.active {{
      background: rgba(239, 68, 68, 0.28);
      border-color: var(--danger);
      box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.18);
    }}
    .fp-label {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      font-weight: 700;
      color: #fecaca;
      cursor: pointer;
      user-select: none;
    }}
    .fp-label input[type="checkbox"] {{
      width: 18px;
      height: 18px;
      accent-color: var(--danger);
      cursor: pointer;
    }}
    .form-footer {{
      padding: 12px 20px;
      border-top: 1px solid var(--border);
      background: var(--panel);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .status-toast {{
      font-size: 12px;
      min-height: 18px;
      color: var(--muted);
    }}
    .status-toast.ok {{ color: #86efac; }}
    .status-toast.err {{ color: #fca5a5; }}
    .modal-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(2, 6, 23, 0.78);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
      z-index: 50;
    }}
    .modal-backdrop.open {{
      display: flex;
    }}
    .modal {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      width: min(980px, 100%);
      max-height: 88vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    .modal-head {{
      padding: 12px 18px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .modal-head h2 {{
      margin: 0;
      font-size: 16px;
    }}
    .modal-body {{
      padding: 16px 18px;
      overflow-y: auto;
      font-size: 13px;
      line-height: 1.5;
    }}
    .modal-body table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 18px;
      font-size: 12px;
    }}
    .modal-body th, .modal-body td {{
      border: 1px solid var(--border);
      padding: 6px 9px;
      text-align: left;
    }}
    .modal-body th {{
      background: var(--panel-2);
    }}
    pre.md-raw {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      overflow-x: auto;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    @media (max-width: 960px) {{
      body {{ height: auto; overflow: auto; }}
      main {{ grid-template-columns: 1fr; }}
      .photo-stage {{ min-height: 340px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>Conferência Rápida QA</h1>
      <span class="file-pill" title="{nome_arquivo}">{nome_arquivo}</span>
    </div>
    <div class="toolbar">
      <div class="progress-wrap">
        <span id="progressText">0 / 0 conferidos</span>
        <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
      </div>
      <div class="segmented" role="group" aria-label="Filtro de registros">
        <button type="button" id="filterAll" class="active">Todos</button>
        <button type="button" id="filterPending">Só pendentes</button>
      </div>
      <button type="button" class="btn btn-primary" id="btnRelatorio">Gerar Relatório QA</button>
    </div>
  </header>

  <main>
    <section class="photo-pane">
      <div class="photo-topbar">
        <span id="fotoPathLabel">Foto: —</span>
        <button type="button" class="btn btn-sm" id="btnZoom">Lupa / Zoom (1x)</button>
      </div>
      <div class="photo-stage" id="photoStage">
        <img id="fotoImg" alt="Foto do veículo detectado" />
        <div class="photo-error" id="fotoError"></div>
      </div>
    </section>

    <section class="form-pane">
      <div class="form-scroll">
        <div class="meta-card">
          <div class="meta-head">
            <span class="nav-index" id="navInfo">Registro 0 de 0</span>
            <div style="display:flex; gap:8px; align-items:center;">
              <select id="jumpSelect" class="btn btn-sm" title="Ir para registro"></select>
              <span class="badge badge-pending" id="statusBadge">Pendente</span>
            </div>
          </div>
          <div class="meta-grid">
            <div class="meta-item"><span>Placa OCR:</span> <strong class="plate-big" id="metaPlaca">—</strong></div>
            <div class="meta-item"><span>Conf. OCR:</span> <strong id="metaConfOcr">—</strong></div>
            <div class="meta-item"><span>Veículo:</span> <strong id="metaTipoCor">—</strong></div>
            <div class="meta-item"><span>VMMR:</span> <strong id="metaConfVmmr">—</strong></div>
            <div class="meta-item"><span>Data/Hora:</span> <strong id="metaDataHora">—</strong></div>
            <div class="meta-item"><span>Vídeo:</span> <strong id="metaVideo">—</strong></div>
          </div>
        </div>

        <label class="fp-banner" id="fpBanner" for="chkFalsoPositivo">
          <span class="fp-label">
            <input type="checkbox" id="chkFalsoPositivo" />
            <span>Falso positivo (sem veículo na foto)</span>
          </span>
          <span class="field-hint" style="color:#fca5a5;">SEM_VEICULO</span>
        </label>

        <div class="field-card">
          <div class="field-header">
            <label for="inputPlacaReal">Placa real</label>
            <span class="field-hint">Uppercase automático</span>
          </div>
          <input type="text" id="inputPlacaReal" class="input-plate" placeholder="Ex: BRA2E19, GWB$$$$, $$$$$$$ ou SEM_PLACA" autocomplete="off" />
          <div class="btn-row">
            <button type="button" class="chip-btn" id="btnPlacaIgual">Placa igual</button>
            <button type="button" class="chip-btn" id="btnSemPlaca">Sem placa</button>
            <button type="button" class="chip-btn" id="btnIlegivel">Ilegível ($$$$$$$)</button>
            <button type="button" class="chip-btn" id="btnCharIlegivel">+ $ (ilegível)</button>
          </div>
          <p style="margin:8px 0 0;font-size:12px;color:var(--muted);">
            Obs.: Use <strong>SEM_PLACA</strong> quando não houver placa visível no enquadramento;
            use <strong>$</strong> nos caracteres incertos (ex.: <code>GWB$$$$</code>);
            e use <strong>$$$$$$$</strong> quando a placa aparecer na foto, mas estiver totalmente ilegível.
          </p>
        </div>

        <div class="field-card">
          <div class="field-header">
            <label for="inputModeloReal">Modelo real</label>
            <span class="field-hint" id="metaMarcaModelo">Top 1: —</span>
          </div>
          <input type="text" id="inputModeloReal" list="catalogoModelos" placeholder="Digite ou escolha o modelo..." autocomplete="off" />
          <datalist id="catalogoModelos">
{opcoes_datalist}
          </datalist>
          <div class="btn-row">
            <button type="button" class="chip-btn" id="btnTop1">Top 1</button>
            <button type="button" class="chip-btn" id="btnTop2">Top 2</button>
            <button type="button" class="chip-btn" id="btnTop3">Top 3</button>
          </div>
        </div>

        <div class="field-card">
          <div class="field-header">
            <label for="inputCorReal">Cor real</label>
            <span class="field-hint" id="metaCor">Estimada: —</span>
          </div>
          <div class="btn-row" id="corButtons">
            <button type="button" class="chip-btn" id="btnCorIgual">Cor igual</button>
{botoes_cor}
          </div>
          <input type="text" id="inputCorReal" placeholder="Digite ou escolha a cor..." autocomplete="off" />
        </div>

        <div class="field-card">
          <div class="field-header">
            <label>Condição</label>
            <span class="field-hint" id="condHint">Memoriza última condição</span>
          </div>
          <div class="btn-row" id="condButtons">
{botoes_condicao}
          </div>
          <input type="text" id="inputCondicao" placeholder="Ou digite outra condição..." autocomplete="off" />
        </div>
      </div>

      <div class="form-footer">
        <button type="button" class="btn" id="btnAnterior">← Anterior</button>
        <span class="status-toast" id="statusToast">Atalho: Ctrl+Enter para Salvar e Próximo</span>
        <button type="button" class="btn btn-success" id="btnSalvarProximo">Salvar e Próximo →</button>
      </div>
    </section>
  </main>

  <div class="modal-backdrop" id="modalRelatorio">
    <div class="modal">
      <div class="modal-head">
        <h2>Relatório de Avaliação QA</h2>
        <div style="display:flex; gap:8px;">
          <button type="button" class="btn btn-sm" id="btnCopiarMd">Copiar Markdown</button>
          <button type="button" class="btn btn-sm" id="btnFecharModal">Fechar</button>
        </div>
      </div>
      <div class="modal-body" id="modalBody">Carregando...</div>
    </div>
  </div>

  <script>
    const STORAGE_KEY_COND = "alpr_qa_ultima_condicao";
    let todosRegistros = [];
    let filtro = "all"; // "all" | "pending"
    let indicesFiltrados = [];
    let posAtual = 0;
    let ultimaCondicao = localStorage.getItem(STORAGE_KEY_COND) || "";
    let ultimoMarkdown = "";
    let salvando = false;

    const el = (id) => document.getElementById(id);

    function atualizarListaFiltrada(manterRow = null) {{
      indicesFiltrados = [];
      todosRegistros.forEach((r, idx) => {{
        if (filtro === "all" || !r.conferido) {{
          indicesFiltrados.push(idx);
        }}
      }});

      if (manterRow !== null) {{
        const pos = indicesFiltrados.findIndex((i) => todosRegistros[i].row === manterRow);
        if (pos !== -1) {{
          posAtual = pos;
          return;
        }}
      }}
      if (posAtual >= indicesFiltrados.length) {{
        posAtual = Math.max(0, indicesFiltrados.length - 1);
      }}
    }}

    function atualizarProgresso() {{
      const total = todosRegistros.length;
      const conferidos = todosRegistros.filter((r) => r.conferido).length;
      const pct = total > 0 ? Math.round((conferidos / total) * 100) : 0;
      el("progressText").textContent = `${{conferidos}} / ${{total}} conferidos (${{pct}}%)`;
      el("progressFill").style.width = `${{pct}}%`;
    }}

    function atualizarSelectSalto() {{
      const sel = el("jumpSelect");
      sel.innerHTML = "";
      indicesFiltrados.forEach((idxReg, pos) => {{
        const r = todosRegistros[idxReg];
        const opt = document.createElement("option");
        opt.value = String(pos);
        const marca = r.conferido ? "✓" : "•";
        opt.textContent = `${{marca}} Linha ${{r.row}} (${{r.placa || "SEM_PLACA"}})`;
        if (pos === posAtual) opt.selected = true;
        sel.appendChild(opt);
      }});
    }}

    function definirCorUI(valor) {{
      el("inputCorReal").value = valor;
      const norm = valor.trim().toLowerCase();
      document.querySelectorAll("#corButtons .chip-btn[data-cor]").forEach((b) => {{
        b.classList.toggle("active", b.dataset.cor.toLowerCase() === norm);
      }});
    }}

    function definirCondicaoUI(valor, memorizada = false) {{
      el("inputCondicao").value = valor;
      document.querySelectorAll("#condButtons .chip-btn").forEach((b) => {{
        b.classList.toggle("active", b.dataset.cond === valor);
      }});
      if (memorizada && valor) {{
        el("condHint").textContent = `Pré-preenchida (última usada: ${{valor}})`;
      }} else {{
        el("condHint").textContent = "Memoriza última condição";
      }}
    }}

    function definirZoom(ativo) {{
      const stage = el("photoStage");
      const img = el("fotoImg");
      stage.classList.toggle("zoomed", ativo);
      el("btnZoom").textContent = ativo ? "Lupa / Zoom (2.6x)" : "Lupa / Zoom (1x)";
      if (!ativo) {{
        img.style.transformOrigin = "center center";
      }}
    }}

    function carregarFoto(row, caminhoFoto) {{
      const img = el("fotoImg");
      const err = el("fotoError");
      definirZoom(false);
      el("fotoPathLabel").textContent = `Foto: ${{caminhoFoto || "não informada"}}`;
      img.style.display = "none";
      err.style.display = "none";
      img.onload = () => {{
        err.style.display = "none";
        img.style.display = "block";
      }};
      img.onerror = async () => {{
        img.style.display = "none";
        let detalhe = caminhoFoto ? `Caminho salvo não encontrado: ${{caminhoFoto}}` : "Sem caminho de foto na planilha.";
        try {{
          const resp = await fetch(`/foto?row=${{row}}`);
          if (!resp.ok) {{
            const body = await resp.json().catch(() => null);
            if (body && body.erro) detalhe = body.erro;
          }}
        }} catch (_) {{}}
        err.textContent = `Foto indisponível (404) — ${{detalhe}}`;
        err.style.display = "block";
      }};
      img.src = `/foto?row=${{row}}&_t=${{Date.now()}}`;
    }}

    function renderizarAtual() {{
      atualizarProgresso();
      atualizarSelectSalto();

      const semItens = indicesFiltrados.length === 0;
      el("btnAnterior").disabled = semItens || posAtual <= 0;
      el("btnSalvarProximo").disabled = semItens;

      if (semItens) {{
        el("navInfo").textContent = filtro === "pending"
          ? "Nenhum registro pendente!"
          : "Nenhum registro na planilha";
        el("statusBadge").textContent = "Concluído";
        el("statusBadge").className = "badge badge-ok";
        el("chkFalsoPositivo").checked = false;
        el("chkFalsoPositivo").disabled = true;
        el("fpBanner").classList.remove("active");
        el("fotoImg").style.display = "none";
        el("fotoError").style.display = "block";
        el("fotoError").textContent = filtro === "pending"
          ? "Todos os registros já foram conferidos. Clique em 'Gerar Relatório QA' ou mude o filtro para 'Todos'."
          : "Planilha vazia.";
        return;
      }}

      const r = todosRegistros[indicesFiltrados[posAtual]];
      el("navInfo").textContent = `Item ${{posAtual + 1}} de ${{indicesFiltrados.length}} (Linha Excel #${{r.row}})`;
      el("statusBadge").textContent = r.conferido ? "Conferido" : "Pendente";
      el("statusBadge").className = `badge ${{r.conferido ? "badge-ok" : "badge-pending"}}`;

      el("metaPlaca").textContent = r.placa || "—";
      el("metaConfOcr").textContent = r.conf_ocr !== "" ? String(r.conf_ocr) : "—";
      el("metaTipoCor").textContent = [r.tipo, r.cor].filter(Boolean).join(" • ") || "—";
      el("metaConfVmmr").textContent = r.confiabilidade || "—";
      el("metaDataHora").textContent = [r.data, r.hora].filter(Boolean).join(" ") || "—";
      el("metaVideo").textContent = r.video || "—";
      el("metaMarcaModelo").textContent = `Sugestão: ${{r.marca_modelo || "—"}}`;

      const ehFalsoPositivo =
        (r.placa_real || "").toUpperCase() === "SEM_VEICULO" ||
        (r.modelo_real || "").toUpperCase() === "SEM_VEICULO";
      el("chkFalsoPositivo").disabled = false;
      el("chkFalsoPositivo").checked = ehFalsoPositivo;
      el("fpBanner").classList.toggle("active", ehFalsoPositivo);

      el("inputPlacaReal").value = r.placa_real || "";
      el("btnPlacaIgual").disabled = !r.placa;
      el("btnPlacaIgual").textContent = r.placa ? `Placa igual (${{r.placa}})` : "Placa igual";

      el("inputModeloReal").value = r.modelo_real || "";

      const configurarTopBtn = (btnId, label, textoRaw, modeloLimpo) => {{
        const btn = el(btnId);
        const valido = Boolean(modeloLimpo);
        btn.disabled = !valido;
        btn.dataset.modelo = modeloLimpo || "";
        btn.textContent = valido ? `${{label}}: ${{textoRaw}}` : `${{label}}: —`;
      }};
      configurarTopBtn("btnTop1", "Top 1", r.desc_1, r.desc_1_modelo);
      configurarTopBtn("btnTop2", "Top 2", r.desc_2, r.desc_2_modelo);
      configurarTopBtn("btnTop3", "Top 3", r.desc_3, r.desc_3_modelo);

      el("metaCor").textContent = `Estimada: ${{r.cor || "—"}}`;
      el("btnCorIgual").disabled = !r.cor;
      el("btnCorIgual").textContent = r.cor ? `Cor igual (${{r.cor}})` : "Cor igual";
      definirCorUI(r.cor_real || "");

      if (r.condicao) {{
        definirCondicaoUI(r.condicao, false);
      }} else if (ultimaCondicao) {{
        definirCondicaoUI(ultimaCondicao, true);
      }} else {{
        definirCondicaoUI("", false);
      }}

      carregarFoto(r.row, r.foto);
    }}

    function mostrarToast(msg, tipo = "") {{
      const toast = el("statusToast");
      toast.textContent = msg;
      toast.className = `status-toast ${{tipo}}`;
    }}

    async function carregarRegistros() {{
      const resp = await fetch("/api/registros");
      if (!resp.ok) throw new Error("Falha ao carregar /api/registros");
      const data = await resp.json();
      todosRegistros = data.registros || [];
      atualizarListaFiltrada();
      if (filtro === "all") {{
        const primeiroPendente = indicesFiltrados.findIndex(
          (idxReg) => !todosRegistros[idxReg].conferido
        );
        posAtual = primeiroPendente !== -1 ? primeiroPendente : 0;
      }}
      renderizarAtual();
    }}

    async function salvarEAvancar() {{
      if (salvando || indicesFiltrados.length === 0) return;
      const idxReg = indicesFiltrados[posAtual];
      const r = todosRegistros[idxReg];
      const placaReal = el("inputPlacaReal").value.trim().toUpperCase();
      const modeloReal = el("inputModeloReal").value.trim();
      const corReal = el("inputCorReal").value.trim();
      const condicao = el("inputCondicao").value.trim();

      el("inputPlacaReal").value = placaReal;

      salvando = true;
      el("btnSalvarProximo").disabled = true;
      mostrarToast("Salvando na planilha...");

      try {{
        const resp = await fetch("/api/salvar", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            row: r.row,
            placa_real: placaReal,
            modelo_real: modeloReal,
            cor_real: corReal,
            condicao: condicao,
          }}),
        }});
        const payload = await resp.json();
        if (!resp.ok) {{
          throw new Error(payload.erro || "Erro ao salvar");
        }}

        r.placa_real = payload.placa_real;
        r.modelo_real = payload.modelo_real;
        r.cor_real = payload.cor_real;
        r.condicao = payload.condicao;
        r.conferido = payload.conferido;

        if (condicao) {{
          ultimaCondicao = condicao;
          localStorage.setItem(STORAGE_KEY_COND, condicao);
        }}

        mostrarToast(`Linha #${{r.row}} salva!`, "ok");

        if (filtro === "pending") {{
          atualizarListaFiltrada();
        }} else if (posAtual < indicesFiltrados.length - 1) {{
          posAtual += 1;
        }}
        renderizarAtual();
      }} catch (err) {{
        mostrarToast(err.message, "err");
      }} finally {{
        salvando = false;
        el("btnSalvarProximo").disabled = indicesFiltrados.length === 0;
      }}
    }}

    function markdownParaHtmlSimples(md) {{
      const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const linhas = md.split(/\\r?\\n/);
      let out = [];
      let emTabela = false;
      let cabecalhoFeito = false;

      const fecharTabela = () => {{
        if (emTabela) {{
          out.push("</tbody></table>");
          emTabela = false;
          cabecalhoFeito = false;
        }}
      }};

      for (const linha of linhas) {{
        const t = linha.trim();
        if (t.startsWith("|") && t.endsWith("|")) {{
          const celulas = t.slice(1, -1).split("|").map((c) => c.trim());
          if (celulas.every((c) => /^:?-{3,}:?$/.test(c))) {{
            continue;
          }}
          if (!emTabela) {{
            out.push("<table><thead><tr>");
            celulas.forEach((c) => out.push(`<th>${{esc(c).replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>")}}</th>`));
            out.push("</tr></thead><tbody>");
            emTabela = true;
            cabecalhoFeito = true;
          }} else {{
            out.push("<tr>");
            celulas.forEach((c) => out.push(`<td>${{esc(c).replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>")}}</td>`));
            out.push("</tr>");
          }}
          continue;
        }}
        fecharTabela();
        if (t.startsWith("# ")) {{
          out.push(`<h3>${{esc(t.slice(2))}}</h3>`);
        }} else if (t.startsWith("## ")) {{
          out.push(`<h4>${{esc(t.slice(3))}}</h4>`);
        }} else if (t.startsWith("> ")) {{
          out.push(`<p style="color:#94a3b8;">${{esc(t.slice(2))}}</p>`);
        }} else if (t) {{
          out.push(`<p>${{esc(t)}}</p>`);
        }}
      }}
      fecharTabela();
      out.push("<h4>Markdown bruto</h4><pre class=\\"md-raw\\">" + esc(md) + "</pre>");
      return out.join("\\n");
    }}

    async function abrirRelatorio() {{
      el("modalRelatorio").classList.add("open");
      el("modalBody").textContent = "Calculando métricas com avaliar_alpr.py...";
      try {{
        const resp = await fetch("/api/avaliar");
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.erro || "Erro ao gerar relatório");
        ultimoMarkdown = data.markdown || "";
        el("modalBody").innerHTML = markdownParaHtmlSimples(ultimoMarkdown);
      }} catch (err) {{
        el("modalBody").innerHTML = `<p style="color:#fca5a5;">${{err.message}}</p>`;
      }}
    }}

    // Eventos UI
    el("filterAll").addEventListener("click", () => {{
      const rowAtual = indicesFiltrados.length ? todosRegistros[indicesFiltrados[posAtual]].row : null;
      filtro = "all";
      el("filterAll").classList.add("active");
      el("filterPending").classList.remove("active");
      atualizarListaFiltrada(rowAtual);
      renderizarAtual();
    }});

    el("filterPending").addEventListener("click", () => {{
      const rowAtual = indicesFiltrados.length ? todosRegistros[indicesFiltrados[posAtual]].row : null;
      filtro = "pending";
      el("filterPending").classList.add("active");
      el("filterAll").classList.remove("active");
      atualizarListaFiltrada(rowAtual);
      renderizarAtual();
    }});

    el("jumpSelect").addEventListener("change", (e) => {{
      posAtual = Number(e.target.value) || 0;
      renderizarAtual();
    }});

    function sincronizarCheckboxFalsoPositivo() {{
      const placa = el("inputPlacaReal").value.trim().toUpperCase();
      const modelo = el("inputModeloReal").value.trim().toUpperCase();
      const ehFp = placa === "SEM_VEICULO" || modelo === "SEM_VEICULO";
      el("chkFalsoPositivo").checked = ehFp;
      el("fpBanner").classList.toggle("active", ehFp);
    }}

    el("chkFalsoPositivo").addEventListener("change", (e) => {{
      const marcado = e.target.checked;
      el("fpBanner").classList.toggle("active", marcado);
      if (marcado) {{
        el("inputPlacaReal").value = "SEM_VEICULO";
        el("inputModeloReal").value = "SEM_VEICULO";
        definirCorUI("-");
      }} else {{
        if (el("inputPlacaReal").value.trim().toUpperCase() === "SEM_VEICULO") {{
          el("inputPlacaReal").value = "";
        }}
        if (el("inputModeloReal").value.trim().toUpperCase() === "SEM_VEICULO") {{
          el("inputModeloReal").value = "";
        }}
        if (el("inputCorReal").value.trim() === "-") {{
          definirCorUI("");
        }}
      }}
    }});

    el("inputPlacaReal").addEventListener("input", (e) => {{
      const pos = e.target.selectionStart;
      e.target.value = e.target.value.toUpperCase();
      if (pos !== null) e.target.setSelectionRange(pos, pos);
      sincronizarCheckboxFalsoPositivo();
    }});

    el("inputModeloReal").addEventListener("input", () => {{
      sincronizarCheckboxFalsoPositivo();
    }});

    el("btnPlacaIgual").addEventListener("click", () => {{
      if (indicesFiltrados.length === 0) return;
      const r = todosRegistros[indicesFiltrados[posAtual]];
      if (r.placa) el("inputPlacaReal").value = r.placa.toUpperCase();
    }});

    el("btnSemPlaca").addEventListener("click", () => {{
      el("inputPlacaReal").value = "SEM_PLACA";
    }});

    el("btnIlegivel").addEventListener("click", () => {{
      el("inputPlacaReal").value = "$$$$$$$";
    }});

    el("btnCharIlegivel").addEventListener("click", () => {{
      const inp = el("inputPlacaReal");
      const start = inp.selectionStart ?? inp.value.length;
      const end = inp.selectionEnd ?? inp.value.length;
      inp.value = inp.value.slice(0, start) + "$" + inp.value.slice(end);
      inp.focus();
      inp.setSelectionRange(start + 1, start + 1);
    }});

    ["btnTop1", "btnTop2", "btnTop3"].forEach((id) => {{
      el(id).addEventListener("click", (e) => {{
        const modelo = e.currentTarget.dataset.modelo || "";
        if (modelo) el("inputModeloReal").value = modelo;
      }});
    }});

    el("btnCorIgual").addEventListener("click", () => {{
      if (indicesFiltrados.length === 0) return;
      const r = todosRegistros[indicesFiltrados[posAtual]];
      if (r.cor) definirCorUI(r.cor);
    }});

    document.querySelectorAll("#corButtons .chip-btn[data-cor]").forEach((btn) => {{
      btn.addEventListener("click", () => {{
        definirCorUI(btn.dataset.cor);
      }});
    }});

    el("inputCorReal").addEventListener("input", (e) => {{
      const norm = e.target.value.trim().toLowerCase();
      document.querySelectorAll("#corButtons .chip-btn[data-cor]").forEach((b) => {{
        b.classList.toggle("active", b.dataset.cor.toLowerCase() === norm);
      }});
    }});

    document.querySelectorAll("#condButtons .chip-btn").forEach((btn) => {{
      btn.addEventListener("click", () => {{
        const cond = btn.dataset.cond;
        ultimaCondicao = cond;
        localStorage.setItem(STORAGE_KEY_COND, cond);
        definirCondicaoUI(cond, false);
      }});
    }});

    el("inputCondicao").addEventListener("input", (e) => {{
      const val = e.target.value.trim();
      document.querySelectorAll("#condButtons .chip-btn").forEach((b) => {{
        b.classList.toggle("active", b.dataset.cond === val);
      }});
    }});

    el("btnAnterior").addEventListener("click", () => {{
      if (posAtual > 0) {{
        posAtual -= 1;
        renderizarAtual();
      }}
    }});

    el("btnSalvarProximo").addEventListener("click", salvarEAvancar);

    document.addEventListener("keydown", (e) => {{
      if (e.ctrlKey && e.key === "Enter") {{
        e.preventDefault();
        salvarEAvancar();
      }}
    }});

    // Zoom / lupa na imagem
    el("fotoImg").addEventListener("click", (e) => {{
      const stage = el("photoStage");
      const vaiAtivar = !stage.classList.contains("zoomed");
      if (vaiAtivar) {{
        const rect = e.target.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        e.target.style.transformOrigin = `${{x.toFixed(1)}}% ${{y.toFixed(1)}}%`;
      }}
      definirZoom(vaiAtivar);
    }});

    el("photoStage").addEventListener("mousemove", (e) => {{
      const stage = el("photoStage");
      if (!stage.classList.contains("zoomed")) return;
      const rect = stage.getBoundingClientRect();
      const x = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
      const y = Math.max(0, Math.min(100, ((e.clientY - rect.top) / rect.height) * 100));
      el("fotoImg").style.transformOrigin = `${{x.toFixed(1)}}% ${{y.toFixed(1)}}%`;
    }});

    el("btnZoom").addEventListener("click", () => {{
      definirZoom(!el("photoStage").classList.contains("zoomed"));
    }});

    el("btnRelatorio").addEventListener("click", abrirRelatorio);
    el("btnFecharModal").addEventListener("click", () => el("modalRelatorio").classList.remove("open"));
    el("modalRelatorio").addEventListener("click", (e) => {{
      if (e.target === el("modalRelatorio")) el("modalRelatorio").classList.remove("open");
    }});
    el("btnCopiarMd").addEventListener("click", async () => {{
      if (!ultimoMarkdown) return;
      await navigator.clipboard.writeText(ultimoMarkdown);
      mostrarToast("Markdown copiado para a área de transferência!", "ok");
    }});

    carregarRegistros().catch((err) => {{
      mostrarToast(err.message, "err");
    }});
  </script>
</body>
</html>
"""


def criar_handler(
    caminho_xlsx: Path,
    aba: str | None = None,
    modelos_catalogo: list[str] | None = None,
    caminho_gabarito: Path | None = None,
):
    """Cria a classe Handler vinculada à planilha e ao catálogo de modelos."""
    catalogo = modelos_catalogo if modelos_catalogo is not None else carregar_catalogo_modelos()
    html_pagina = _construir_html(caminho_xlsx, catalogo).encode("utf-8")

    class ConferenciaHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Silencia logs ruidosos por request no terminal
            return

        def _enviar_json(self, status: int, payload: dict[str, Any]) -> None:
            corpo = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corpo)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            rota = parsed.path

            if rota == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_pagina)))
                self.end_headers()
                self.wfile.write(html_pagina)
                return

            if rota == "/api/registros":
                try:
                    registros = carregar_registros_planilha(
                        caminho_xlsx, aba=aba, caminho_gabarito=caminho_gabarito
                    )
                    conferidos = sum(1 for r in registros if r["conferido"])
                    self._enviar_json(
                        HTTPStatus.OK,
                        {
                            "registros": registros,
                            "total": len(registros),
                            "conferidos": conferidos,
                            "arquivo": str(caminho_xlsx),
                        },
                    )
                except Exception as exc:
                    self._enviar_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"erro": str(exc)})
                return

            if rota == "/foto":
                qs = parse_qs(parsed.query)
                row_str = (qs.get("row") or [""])[0]
                try:
                    row_num = int(row_str)
                except ValueError:
                    self._enviar_json(
                        HTTPStatus.BAD_REQUEST,
                        {"erro": "Parâmetro 'row' inválido."},
                    )
                    return

                caminho_foto, foto_raw = resolver_caminho_foto(caminho_xlsx, row_num, aba=aba)
                if caminho_foto is None:
                    msg = (
                        f"Arquivo de foto não encontrado no disco: {foto_raw}"
                        if foto_raw
                        else f"Nenhuma foto informada na linha {row_num}."
                    )
                    self._enviar_json(HTTPStatus.NOT_FOUND, {"erro": msg})
                    return

                mime, _ = mimetypes.guess_type(str(caminho_foto))
                dados = caminho_foto.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime or "image/jpeg")
                self.send_header("Content-Length", str(len(dados)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(dados)
                return

            if rota == "/api/avaliar":
                try:
                    md = gerar_relatorio_qa(
                        caminho_xlsx, aba=aba, caminho_gabarito=caminho_gabarito
                    )
                    self._enviar_json(HTTPStatus.OK, {"markdown": md})
                except Exception as exc:
                    self._enviar_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"erro": str(exc)})
                return

            self._enviar_json(HTTPStatus.NOT_FOUND, {"erro": "Rota não encontrada."})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/salvar":
                self._enviar_json(HTTPStatus.NOT_FOUND, {"erro": "Rota não encontrada."})
                return

            try:
                tamanho = int(self.headers.get("Content-Length", "0"))
                bruto = self.rfile.read(tamanho)
                dados = json.loads(bruto.decode("utf-8"))
                row_num = int(dados["row"])
            except Exception:
                self._enviar_json(
                    HTTPStatus.BAD_REQUEST,
                    {"erro": "Payload JSON inválido. Esperado: {row, placa_real, modelo_real, cor_real, condicao}."},
                )
                return

            try:
                salvo = salvar_registro(
                    caminho_xlsx=caminho_xlsx,
                    row=row_num,
                    placa_real=dados.get("placa_real", ""),
                    modelo_real=dados.get("modelo_real", ""),
                    condicao=dados.get("condicao", ""),
                    cor_real=dados.get("cor_real", ""),
                    aba=aba,
                    caminho_gabarito=caminho_gabarito,
                )
                self._enviar_json(HTTPStatus.OK, salvo)
            except PermissionError:
                self._enviar_json(
                    HTTPStatus.LOCKED,
                    {"erro": "Planilha aberta no Excel. Feche o arquivo para salvar."},
                )
            except ValueError as exc:
                self._enviar_json(HTTPStatus.BAD_REQUEST, {"erro": str(exc)})
            except Exception as exc:
                self._enviar_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"erro": str(exc)})

    return ConferenciaHandler


def criar_servidor(
    caminho_xlsx: Path,
    porta: int = 8765,
    aba: str | None = None,
    modelos_catalogo: list[str] | None = None,
    caminho_gabarito: Path | None = None,
) -> ThreadingHTTPServer:
    """Instancia o servidor HTTP local para a planilha informada."""
    handler = criar_handler(
        caminho_xlsx,
        aba=aba,
        modelos_catalogo=modelos_catalogo,
        caminho_gabarito=caminho_gabarito,
    )
    return ThreadingHTTPServer(("127.0.0.1", porta), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Painel web local para conferência rápida da planilha ALPR & VMMR."
    )
    parser.add_argument(
        "caminho_xlsx",
        nargs="?",
        type=Path,
        default=Path("data/placas_detectadas.xlsx"),
        help="Caminho da planilha .xlsx (padrão: data/placas_detectadas.xlsx)",
    )
    parser.add_argument(
        "--porta",
        type=int,
        default=8765,
        help="Porta HTTP local (padrão: 8765)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Não abrir o navegador automaticamente ao iniciar",
    )
    parser.add_argument(
        "--aba",
        type=str,
        default=None,
        help="Nome da aba da planilha (padrão: 'Placas' se existir, senão aba ativa)",
    )

    args = parser.parse_args(argv)
    if not args.caminho_xlsx.exists():
        print(f"Erro: arquivo não encontrado: {args.caminho_xlsx}", file=sys.stderr)
        return 1

    try:
        garantir_colunas_gabarito(args.caminho_xlsx, aba=args.aba)
        restaurados = restaurar_gabarito_na_planilha(args.caminho_xlsx, aba=args.aba)
        if restaurados:
            print(
                f"Gabarito restaurado automaticamente em {restaurados} linha(s) a partir de gabarito_qa.json."
            )
    except PermissionError:
        print(
            f"Aviso: '{args.caminho_xlsx}' está aberto no Excel; feche-o antes de salvar alterações.",
            file=sys.stderr,
        )

    modelos = carregar_catalogo_modelos()
    servidor = criar_servidor(
        caminho_xlsx=args.caminho_xlsx,
        porta=args.porta,
        aba=args.aba,
        modelos_catalogo=modelos,
    )
    url = f"http://127.0.0.1:{servidor.server_port}/"
    print(f"Painel de conferência disponível em: {url}")
    print(f"Planilha: {args.caminho_xlsx} | Modelos no catálogo: {len(modelos)}")
    print("Pressione Ctrl+C para encerrar.")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor de conferência.")
    finally:
        servidor.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
