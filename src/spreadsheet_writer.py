"""
Grava as placas detectadas numa planilha .xlsx, criando o arquivo com
cabeçalho se ainda não existir, ou apendando linhas se já existir.
"""

import os
import sys
from datetime import datetime
from openpyxl import Workbook, load_workbook

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

COLUNAS = [
    "Placa",
    "Tipo",
    "Cor",
    "Marca / Modelo",
    "Descoberta 1 (%)",
    "Descoberta 2 (%)",
    "Descoberta 3 (%)",
    "Confiabilidade VMMR",
    "Data",
    "Hora",
    "Confiança OCR",
    "Foto",
    "Vídeo de origem",
]


class PlanilhaWriter:
    def __init__(self):
        self.caminho = config.PLANILHA_PATH
        self._garantir_arquivo()

    def _migrar_estrutura(self, ws) -> bool:
        """
        Adapta retrocompativelmente planilhas com esquemas legados para o novo padrão:
        ["Placa", "Tipo", "Cor", "Marca / Modelo", "Descoberta 1 (%)", "Descoberta 2 (%)",
         "Descoberta 3 (%)", "Confiabilidade VMMR", "Data", "Hora", "Confiança OCR", "Foto", "Vídeo de origem"]
        """
        if ws.cell(1, 5).value == "Descoberta 1 (%)":
            return False

        alterado = False

        # Caso legado 1: Cabeçalho com "Top 3 Descobertas" na coluna 5
        if ws.cell(1, 5).value == "Top 3 Descobertas":
            ws.insert_cols(6, 2)
            ws.cell(1, 5).value = "Descoberta 1 (%)"
            ws.cell(1, 6).value = "Descoberta 2 (%)"
            ws.cell(1, 7).value = "Descoberta 3 (%)"
            ws.cell(1, 8).value = "Confiabilidade VMMR"

            for r in range(2, ws.max_row + 1):
                val_top3 = ws.cell(r, 5).value
                if val_top3 is None or str(val_top3).strip() in ("", "-"):
                    ws.cell(r, 5).value = "-"
                    ws.cell(r, 6).value = "-"
                    ws.cell(r, 7).value = "-"
                else:
                    partes = [p.strip() for p in str(val_top3).split(",") if p.strip()]
                    ws.cell(r, 5).value = partes[0] if len(partes) > 0 else "-"
                    ws.cell(r, 6).value = partes[1] if len(partes) > 1 else "-"
                    ws.cell(r, 7).value = partes[2] if len(partes) > 2 else "-"

                if ws.cell(r, 8).value is None:
                    ws.cell(r, 8).value = "-"

            alterado = True

        # Caso legado 2: Versão com "Data" na coluna 5 (sem VMMR/descobertas)
        elif ws.cell(1, 5).value == "Data":
            ws.insert_cols(5, 4)
            ws.cell(1, 5).value = "Descoberta 1 (%)"
            ws.cell(1, 6).value = "Descoberta 2 (%)"
            ws.cell(1, 7).value = "Descoberta 3 (%)"
            ws.cell(1, 8).value = "Confiabilidade VMMR"
            for r in range(2, ws.max_row + 1):
                for c in range(5, 9):
                    if ws.cell(r, c).value is None:
                        ws.cell(r, c).value = "-"
            alterado = True

        # Caso legado 3: Versão primitiva ("Data" na coluna 2)
        elif ws.cell(1, 2).value == "Data":
            for col_idx, col_nome in enumerate(COLUNAS, start=1):
                ws.cell(1, col_idx, col_nome)
            alterado = True

        return alterado

    def _garantir_arquivo(self):
        os.makedirs(os.path.dirname(self.caminho), exist_ok=True)
        if not os.path.exists(self.caminho):
            wb = Workbook()
            ws = wb.active
            ws.title = "Placas"
            ws.append(COLUNAS)
            wb.save(self.caminho)
        else:
            try:
                wb = load_workbook(self.caminho)
                ws = wb["Placas"]
                if self._migrar_estrutura(ws):
                    wb.save(self.caminho)
            except Exception as e:
                print(f"[PlanilhaWriter] Aviso ao migrar planilha existente: {e}")

    def registrar(self, placa: str, tipo: str, cor: str, marca_modelo: str,
                  dt: datetime, confianca_ocr: float, caminho_foto: str, video_origem: str,
                  desc_1: str = "-", desc_2: str = "-", desc_3: str = "-",
                  confiabilidade: str = "-", top3: str | None = None):
        # Retrocompatibilidade caso top3 seja passado como argumento único
        if top3 is not None and desc_1 == "-" and desc_2 == "-" and desc_3 == "-":
            if isinstance(top3, list):
                partes = [str(x).strip() for x in top3]
            else:
                partes = [p.strip() for p in str(top3).split(",") if p.strip()]
            desc_1 = partes[0] if len(partes) > 0 else "-"
            desc_2 = partes[1] if len(partes) > 1 else "-"
            desc_3 = partes[2] if len(partes) > 2 else "-"

        linha = [
            placa,
            tipo,
            cor,
            marca_modelo,
            str(desc_1) if desc_1 is not None else "-",
            str(desc_2) if desc_2 is not None else "-",
            str(desc_3) if desc_3 is not None else "-",
            confiabilidade,
            dt.strftime("%d/%m/%Y"),
            dt.strftime("%H:%M:%S"),
            round(confianca_ocr, 2),
            caminho_foto,
            video_origem,
        ]

        # Tenta salvar no caminho padrão; se o arquivo estiver aberto no Excel, usa fallback
        caminhos_tentativa = [
            self.caminho,
            os.path.join(os.path.dirname(self.caminho), "placas_detectadas_atualizado.xlsx"),
            os.path.join(os.path.dirname(self.caminho), f"placas_detectadas_{dt.strftime('%H%M%S')}.xlsx"),
        ]

        for cam in caminhos_tentativa:
            try:
                if os.path.exists(cam):
                    wb = load_workbook(cam)
                    ws = wb["Placas"]
                    self._migrar_estrutura(ws)
                else:
                    wb = Workbook()
                    ws = wb.active
                    ws.title = "Placas"
                    ws.append(COLUNAS)

                ws.append(linha)
                wb.save(cam)
                if cam != self.caminho:
                    print(f"[PlanilhaWriter] AVISO: A planilha principal estava aberta no Excel. Salvo em: {cam}")
                break
            except PermissionError:
                continue
