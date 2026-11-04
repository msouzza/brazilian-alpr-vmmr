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

COLUNAS = ["Placa", "Tipo", "Cor", "Marca / Modelo", "Data", "Hora", "Confiança OCR", "Foto", "Vídeo de origem"]


class PlanilhaWriter:
    def __init__(self):
        self.caminho = config.PLANILHA_PATH
        self._garantir_arquivo()

    def _garantir_arquivo(self):
        os.makedirs(os.path.dirname(self.caminho), exist_ok=True)
        if not os.path.exists(self.caminho):
            wb = Workbook()
            ws = wb.active
            ws.title = "Placas"
            ws.append(COLUNAS)
            wb.save(self.caminho)

    def registrar(self, placa: str, tipo: str, cor: str, marca_modelo: str,
                  dt: datetime, confianca_ocr: float, caminho_foto: str, video_origem: str):
        linha = [
            placa,
            tipo,
            cor,
            marca_modelo,
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
                    # Atualiza o cabeçalho se ainda estava no modelo antigo de 6 colunas
                    if ws.cell(1, 2).value == "Data":
                        for col_idx, col_nome in enumerate(COLUNAS, start=1):
                            ws.cell(1, col_idx, col_nome)
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
