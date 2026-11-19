"""
Testes automatizados para o painel de conferência rápida conferir.py.
"""

import io
import json

import pytest
from openpyxl import Workbook, load_workbook

import avaliar_alpr
import conferir
from src.spreadsheet_writer import COLUNAS


def _criar_planilha_sem_gabarito(caminho, linhas, nome_aba="Placas"):
    wb = Workbook()
    ws = wb.active
    ws.title = nome_aba
    ws.append(list(COLUNAS))
    for linha in linhas:
        ws.append(linha)
    wb.save(caminho)


def _linha_exemplo(
    placa="ABC1D23",
    d1="1º Volkswagen Fox (73.5%)",
    d2="2º Hyundai HB20 (12.0%)",
    d3="3º Fiat Argo (5.0%)",
    conf="Alta",
    foto="data/fotos/exemplo.jpg",
    video="video_teste.mp4",
):
    return [
        placa,
        "Carro",
        "Prata / Cinza",
        "Volkswagen Fox",
        d1,
        d2,
        d3,
        conf,
        "18/11/2026",
        "14:30:00",
        0.92,
        foto,
        video,
    ]


def test_garantir_colunas_gabarito_adiciona_sem_sobrescrever_e_idempotente(tmp_path):
    xlsx = tmp_path / "placas.xlsx"
    _criar_planilha_sem_gabarito(xlsx, [_linha_exemplo()])

    indices = conferir.garantir_colunas_gabarito(xlsx)
    assert set(indices.keys()) == {"Placa real", "Modelo real", "Cor real", "Condição"}
    assert indices["Placa real"] == len(COLUNAS) + 1
    assert indices["Modelo real"] == len(COLUNAS) + 2
    assert indices["Cor real"] == len(COLUNAS) + 3
    assert indices["Condição"] == len(COLUNAS) + 4

    wb = load_workbook(xlsx)
    ws = wb["Placas"]
    cabecalho = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    assert cabecalho == list(COLUNAS) + ["Placa real", "Modelo real", "Cor real", "Condição"]
    assert ws.cell(2, 1).value == "ABC1D23"
    assert ws.cell(2, 13).value == "video_teste.mp4"

    # Segunda chamada não duplica colunas
    indices_2 = conferir.garantir_colunas_gabarito(xlsx)
    assert indices_2 == indices
    wb2 = load_workbook(xlsx)
    assert wb2["Placas"].max_column == len(COLUNAS) + 4


def test_garantir_colunas_adiciona_cor_real_em_planilha_legada_preservando_linhas(tmp_path):
    xlsx = tmp_path / "legada.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Placas"
    ws.append(list(COLUNAS) + ["Placa real", "Modelo real", "Condição"])
    ws.append(_linha_exemplo(placa="BRA2E19") + ["BRA2E19", "Volkswagen Fox", "dia_traseira"])
    wb.save(xlsx)

    indices = conferir.garantir_colunas_gabarito(xlsx)
    assert set(indices.keys()) == {"Placa real", "Modelo real", "Cor real", "Condição"}
    assert indices["Cor real"] == len(COLUNAS) + 4

    registros = conferir.carregar_registros_planilha(xlsx)
    assert len(registros) == 1
    assert registros[0]["placa_real"] == "BRA2E19"
    assert registros[0]["modelo_real"] == "Volkswagen Fox"
    assert registros[0]["cor_real"] == ""
    assert registros[0]["condicao"] == "dia_traseira"
    assert registros[0]["conferido"] is True


def test_garantir_colunas_fallback_aba_ativa(tmp_path):
    xlsx = tmp_path / "outra_aba.xlsx"
    _criar_planilha_sem_gabarito(xlsx, [_linha_exemplo()], nome_aba="Resultados")

    indices = conferir.garantir_colunas_gabarito(xlsx)
    assert indices["Placa real"] == len(COLUNAS) + 1

    registros = conferir.carregar_registros_planilha(xlsx)
    assert len(registros) == 1
    assert registros[0]["placa"] == "ABC1D23"


def test_carregar_registros_preserva_row_e_pula_linhas_vazias(tmp_path):
    xlsx = tmp_path / "com_vazia.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Placas"
    ws.append(list(COLUNAS))
    ws.append(_linha_exemplo(placa="AAA1111"))
    ws.append([None] * len(COLUNAS))  # linha 3 vazia
    ws.append(_linha_exemplo(placa="BBB2C33"))
    wb.save(xlsx)

    registros = conferir.carregar_registros_planilha(xlsx)
    assert len(registros) == 2
    assert registros[0]["row"] == 2
    assert registros[0]["placa"] == "AAA1111"
    assert registros[0]["desc_1_modelo"] == "Volkswagen Fox"
    assert registros[0]["desc_2_modelo"] == "Hyundai HB20"
    assert registros[0]["desc_3_modelo"] == "Fiat Argo"
    assert registros[0]["conferido"] is False

    assert registros[1]["row"] == 4
    assert registros[1]["placa"] == "BBB2C33"


def test_salvar_registro_e_compatibilidade_avaliar_alpr(tmp_path):
    xlsx = tmp_path / "qa.xlsx"
    linhas = [
        _linha_exemplo(
            placa="BRA2E19",
            d1="1º Volkswagen Fox (73.5%)",
            d2="2º Hyundai HB20 (12.0%)",
            d3="3º Fiat Argo (5.0%)",
            conf="Alta",
        ),
        _linha_exemplo(
            placa="ELU9D76",
            d1="1º Fiat Palio (45.0%)",
            d2="2º Citroën C3 (35.0%)",
            d3="3º Chevrolet Onix (10.0%)",
            conf="Média",
        ),
        _linha_exemplo(
            placa="SEM_PLACA",
            d1="1º Toyota Corolla (50.0%)",
            d2="2º Honda Civic (30.0%)",
            d3="3º Nissan Sentra (10.0%)",
            conf="Baixa / Ambígua",
        ),
    ]
    _criar_planilha_sem_gabarito(xlsx, linhas)

    r1 = conferir.salvar_registro(
        xlsx,
        row=2,
        placa_real="bra2e19",
        modelo_real="Volkswagen Fox",
        cor_real="Prata / Cinza",
        condicao="dia_traseira",
    )
    assert r1["placa_real"] == "BRA2E19"
    assert r1["cor_real"] == "Prata / Cinza"
    assert r1["conferido"] is True

    conferir.salvar_registro(
        xlsx,
        row=3,
        placa_real="ELU9076",  # 1 erro Hamming
        modelo_real="citroen c3",
        cor_real="Branco",
        condicao="dia_traseira",
    )
    conferir.salvar_registro(
        xlsx,
        row=4,
        placa_real="SEM_PLACA",
        modelo_real="Honda Civic",
        condicao="noite_traseira",
    )

    with pytest.raises(ValueError):
        conferir.salvar_registro(xlsx, row=1, placa_real="X")
    with pytest.raises(ValueError):
        conferir.salvar_registro(xlsx, row=99, placa_real="X")

    # Verifica leitura após salvamento
    lidos = conferir.carregar_registros_planilha(xlsx)
    assert all(item["conferido"] for item in lidos)
    assert lidos[0]["placa_real"] == "BRA2E19"
    assert lidos[0]["cor_real"] == "Prata / Cinza"
    assert lidos[1]["condicao"] == "dia_traseira"
    assert lidos[1]["cor_real"] == "Branco"
    assert lidos[2]["placa_real"] == "SEM_PLACA"
    assert lidos[2]["cor_real"] == ""

    # Valida integração total com avaliar_alpr.py
    aval = avaliar_alpr.carregar_registros(xlsx)
    assert len(aval) == 3
    assert aval[0].placa_exata is True
    assert aval[0].modelo_top1_acerto is True
    assert aval[0].cor_correta is True
    assert aval[1].placa_ate_1_erro is True
    assert aval[1].modelo_top3_acerto is True
    assert aval[1].cor_correta is False
    assert aval[2].sem_placa_legivel is True
    assert aval[2].modelo_top3_acerto is True

    relatorio = conferir.gerar_relatorio_qa(xlsx)
    assert "dia_traseira" in relatorio
    assert "noite_traseira" in relatorio
    assert "Total Geral" in relatorio


class _FakeSocket:
    def __init__(self, raw_request: bytes):
        self._rfile = io.BytesIO(raw_request)
        self._wfile = io.BytesIO()

    def makefile(self, mode: str, *args, **kwargs):
        if "r" in mode:
            return self._rfile
        return self._wfile

    def sendall(self, data: bytes):
        self._wfile.write(data)

    def close(self):
        pass


def _invocar_handler(handler_cls, method: str, path: str, body: dict | None = None):
    body_bytes = b""
    headers = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        headers += (
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
        )
    raw_req = headers.encode("ascii") + b"\r\n" + body_bytes
    sock = _FakeSocket(raw_req)
    handler_cls(sock, ("127.0.0.1", 0), None)
    raw_resp = sock._wfile.getvalue()
    head, payload = raw_resp.split(b"\r\n\r\n", 1)
    status_line = head.split(b"\r\n", 1)[0].decode("ascii")
    status_code = int(status_line.split(" ", 2)[1])
    return status_code, head.decode("latin1"), payload


def test_servidor_http_endpoints(tmp_path):
    img_path = tmp_path / "veiculo.jpg"
    img_bytes = b"\xff\xd8\xff\xe0FAKEJPEGDATA"
    img_path.write_bytes(img_bytes)

    xlsx = tmp_path / "srv.xlsx"
    _criar_planilha_sem_gabarito(
        xlsx,
        [
            _linha_exemplo(placa="ABC1D23", foto=str(img_path)),
            _linha_exemplo(placa="XYZ9876", foto=str(tmp_path / "nao_existe.jpg")),
        ],
    )
    conferir.garantir_colunas_gabarito(xlsx)

    handler_cls = conferir.criar_handler(
        xlsx,
        modelos_catalogo=["Volkswagen Fox", "Hyundai HB20"],
    )

    # GET /
    status, _, body = _invocar_handler(handler_cls, "GET", "/")
    assert status == 200
    html_body = body.decode("utf-8")
    assert '<datalist id="catalogoModelos">' in html_body
    assert "Volkswagen Fox" in html_body
    assert "dia_traseira" in html_body
    assert 'id="btnSemPlaca">Sem placa</button>' in html_body
    assert 'id="btnIlegivel">Ilegível ($$$$$$$)</button>' in html_body
    assert 'id="btnCharIlegivel">' in html_body
    assert 'id="inputCorReal"' in html_body
    assert 'id="btnCorIgual"' in html_body
    assert 'data-cor="Prata / Cinza"' in html_body
    assert 'data-cor="Amarelo / Laranja"' in html_body
    assert 'data-cor="Roxo / Vinho"' in html_body

    # GET /api/registros
    status, _, body = _invocar_handler(handler_cls, "GET", "/api/registros")
    assert status == 200
    dados = json.loads(body.decode("utf-8"))
    assert dados["total"] == 2
    assert dados["conferidos"] == 0
    assert dados["registros"][0]["row"] == 2
    assert "cor_real" in dados["registros"][0]

    # GET /foto?row=2 (existe)
    status, _, body = _invocar_handler(handler_cls, "GET", "/foto?row=2")
    assert status == 200
    assert body == img_bytes

    # GET /foto?row=3 (inexistente -> 404 claro)
    status, _, body = _invocar_handler(handler_cls, "GET", "/foto?row=3")
    assert status == 404
    assert "erro" in json.loads(body.decode("utf-8"))

    # POST /api/salvar
    status, _, body = _invocar_handler(
        handler_cls,
        "POST",
        "/api/salvar",
        body={
            "row": 2,
            "placa_real": "ABC1D23",
            "modelo_real": "Volkswagen Fox",
            "cor_real": "Prata / Cinza",
            "condicao": "dia_traseira",
        },
    )
    assert status == 200
    salvo = json.loads(body.decode("utf-8"))
    assert salvo["conferido"] is True
    assert salvo["cor_real"] == "Prata / Cinza"

    # GET /api/avaliar
    status, _, body = _invocar_handler(handler_cls, "GET", "/api/avaliar")
    assert status == 200
    aval_json = json.loads(body.decode("utf-8"))
    assert "Relatório de Avaliação ALPR & VMMR" in aval_json["markdown"]
    assert "dia_traseira" in aval_json["markdown"]


def test_salvar_registro_atualiza_gabarito_json_e_chave_secundaria(tmp_path):
    xlsx = tmp_path / "qa_backup.xlsx"
    _criar_planilha_sem_gabarito(
        xlsx,
        [
            _linha_exemplo(
                placa="BRA2E19",
                foto=r"E:\outro\dir\BRA2E19_2026-07-22.jpg",
                video="vid_01.mp4",
            ),
            _linha_exemplo(
                placa="SEM_PLACA",
                foto="-",
                video="vid_02.mp4",
            ),
        ],
    )

    conferir.salvar_registro(
        xlsx,
        row=2,
        placa_real="bra2e19",
        modelo_real="Volkswagen Fox",
        cor_real="Branco",
        condicao="dia_traseira",
    )
    conferir.salvar_registro(
        xlsx,
        row=3,
        placa_real="SEM_PLACA",
        modelo_real="Hyundai HB20",
        condicao="noite_traseira",
    )

    gabarito_path = tmp_path / "gabarito_qa.json"
    assert gabarito_path.exists()
    dados = json.loads(gabarito_path.read_text(encoding="utf-8"))

    assert "BRA2E19_2026-07-22.jpg" in dados
    assert dados["BRA2E19_2026-07-22.jpg"] == {
        "placa_real": "BRA2E19",
        "modelo_real": "Volkswagen Fox",
        "cor_real": "Branco",
        "condicao": "dia_traseira",
        "video": "vid_01.mp4",
        "data": "18/11/2026",
        "hora": "14:30:00",
        "foto_nome": "BRA2E19_2026-07-22.jpg",
    }

    chave_sec = "vid_02.mp4|18/11/2026|14:30:00"
    assert chave_sec in dados
    assert dados[chave_sec]["foto_nome"] == ""
    assert dados[chave_sec]["modelo_real"] == "Hyundai HB20"

    # Limpar gabarito da linha 3 remove do JSON para não restaurar indevidamente
    conferir.salvar_registro(xlsx, row=3, placa_real="", modelo_real="", condicao="")
    dados_apos = json.loads(gabarito_path.read_text(encoding="utf-8"))
    assert chave_sec not in dados_apos
    assert "BRA2E19_2026-07-22.jpg" in dados_apos


def test_restauracao_automatica_em_nova_planilha_por_nome_foto(tmp_path):
    xlsx_antiga = tmp_path / "placas_antiga.xlsx"
    gabarito_custom = tmp_path / "gabarito_qa.json"
    _criar_planilha_sem_gabarito(
        xlsx_antiga,
        [
            _linha_exemplo(
                placa="BRA2E19",
                d1="1º Volkswagen Fox (73.5%)",
                foto="data/fotos/BRA2E19_frame10.jpg",
                video="L_001.mp4",
            ),
            _linha_exemplo(
                placa="XYZ1A23",
                d1="1º Hyundai HB20 (60.0%)",
                foto="data/fotos/XYZ1A23_frame20.jpg",
                video="L_001.mp4",
            ),
        ],
    )

    conferir.salvar_registro(
        xlsx_antiga,
        row=2,
        placa_real="BRA2E19",
        modelo_real="Volkswagen Fox",
        cor_real="Prata / Cinza",
        condicao="dia_traseira",
        caminho_gabarito=gabarito_custom,
    )

    # Gera nova planilha do zero (sem colunas de gabarito e com caminho de foto diferente, mesmo nome)
    xlsx_nova = tmp_path / "placas_nova.xlsx"
    _criar_planilha_sem_gabarito(
        xlsx_nova,
        [
            _linha_exemplo(
                placa="BRA2E19",
                d1="1º Volkswagen Fox (73.5%)",
                foto="C:/novo_workspace/fotos/BRA2E19_frame10.jpg",
                video="L_001.mp4",
            ),
            _linha_exemplo(
                placa="XYZ1A23",
                d1="1º Hyundai HB20 (60.0%)",
                foto="C:/novo_workspace/fotos/XYZ1A23_frame20.jpg",
                video="L_001.mp4",
            ),
        ],
    )

    registros = conferir.carregar_registros_planilha(xlsx_nova)
    assert len(registros) == 2
    assert registros[0]["conferido"] is True
    assert registros[0]["placa_real"] == "BRA2E19"
    assert registros[0]["modelo_real"] == "Volkswagen Fox"
    assert registros[0]["cor_real"] == "Prata / Cinza"
    assert registros[0]["condicao"] == "dia_traseira"
    assert registros[1]["conferido"] is False

    # Confirma que os dados foram persistidos na planilha nova e visíveis para avaliar_alpr.py
    aval = avaliar_alpr.carregar_registros(xlsx_nova)
    assert len(aval) == 2
    assert aval[0].placa_real == "BRA2E19"
    assert aval[0].modelo_real == "volkswagen fox"
    assert aval[0].cor_real == "prata / cinza"
    assert aval[0].condicao == "dia_traseira"
    assert aval[0].placa_exata is True
    assert aval[0].modelo_top1_acerto is True
    assert aval[0].cor_correta is True

    # Linha já preenchida manualmente não deve ser sobrescrita pelo JSON
    conferir.salvar_registro(
        xlsx_nova,
        row=2,
        placa_real="OUTRA12",
        modelo_real="Fiat Uno",
        condicao="noite_frontal",
        caminho_gabarito=tmp_path / "outro_gabarito.json",
    )
    restaurados = conferir.restaurar_gabarito_na_planilha(
        xlsx_nova, caminho_gabarito=gabarito_custom
    )
    assert restaurados == 0
    regs_pos = conferir.carregar_registros_planilha(
        xlsx_nova, caminho_gabarito=gabarito_custom
    )
    assert regs_pos[0]["placa_real"] == "OUTRA12"
    assert regs_pos[0]["modelo_real"] == "Fiat Uno"


def test_html_contem_autoposicionamento_primeiro_pendente(tmp_path):
    xlsx = tmp_path / "ui.xlsx"
    _criar_planilha_sem_gabarito(xlsx, [_linha_exemplo()])
    handler_cls = conferir.criar_handler(xlsx, modelos_catalogo=["Volkswagen Fox"])
    status, _, body = _invocar_handler(handler_cls, "GET", "/")
    assert status == 200
    html_body = body.decode("utf-8")
    assert "primeiroPendente" in html_body
    assert '!todosRegistros[idxReg].conferido' in html_body


def test_falso_positivo_checkbox_html_e_persistencia(tmp_path):
    xlsx = tmp_path / "fp.xlsx"
    _criar_planilha_sem_gabarito(
        xlsx,
        [
            _linha_exemplo(
                placa="SEM_PLACA",
                foto="data/fotos/falso_positivo_1.jpg",
                video="vid_fp.mp4",
            ),
            _linha_exemplo(
                placa="ABC1D23",
                foto="data/fotos/carro_real.jpg",
                video="vid_fp.mp4",
            ),
        ],
    )
    conferir.garantir_colunas_gabarito(xlsx)

    handler_cls = conferir.criar_handler(xlsx, modelos_catalogo=["Volkswagen Fox"])

    # HTML possui checkbox chkFalsoPositivo e handlers de SEM_VEICULO
    status, _, body = _invocar_handler(handler_cls, "GET", "/")
    assert status == 200
    html_body = body.decode("utf-8")
    assert 'id="chkFalsoPositivo"' in html_body
    assert "Falso positivo (sem veículo na foto)" in html_body
    assert "SEM_VEICULO" in html_body
    assert 'r.placa_real || "").toUpperCase() === "SEM_VEICULO"' in html_body

    # Salva via endpoint /api/salvar como falso positivo
    status, _, body = _invocar_handler(
        handler_cls,
        "POST",
        "/api/salvar",
        body={
            "row": 2,
            "placa_real": "SEM_VEICULO",
            "modelo_real": "SEM_VEICULO",
            "cor_real": "-",
            "condicao": "dia_frontal",
        },
    )
    assert status == 200
    salvo = json.loads(body.decode("utf-8"))
    assert salvo["placa_real"] == "SEM_VEICULO"
    assert salvo["modelo_real"] == "SEM_VEICULO"
    assert salvo["cor_real"] == "-"
    assert salvo["conferido"] is True

    # Persiste no gabarito_qa.json
    gabarito_path = tmp_path / "gabarito_qa.json"
    assert gabarito_path.exists()
    gabarito = json.loads(gabarito_path.read_text(encoding="utf-8"))
    assert gabarito["falso_positivo_1.jpg"]["placa_real"] == "SEM_VEICULO"
    assert gabarito["falso_positivo_1.jpg"]["modelo_real"] == "SEM_VEICULO"
    assert gabarito["falso_positivo_1.jpg"]["cor_real"] == "-"

    # Recarrega da planilha e verifica restauração em nova planilha
    regs = conferir.carregar_registros_planilha(xlsx)
    assert regs[0]["placa_real"] == "SEM_VEICULO"
    assert regs[0]["modelo_real"] == "SEM_VEICULO"
    assert regs[0]["cor_real"] == "-"
    assert regs[0]["conferido"] is True

    xlsx_nova = tmp_path / "fp_nova.xlsx"
    _criar_planilha_sem_gabarito(
        xlsx_nova,
        [
            _linha_exemplo(
                placa="SEM_PLACA",
                foto="outro/caminho/falso_positivo_1.jpg",
                video="vid_fp.mp4",
            ),
        ],
    )
    regs_restaurados = conferir.carregar_registros_planilha(
        xlsx_nova, caminho_gabarito=gabarito_path
    )
    assert regs_restaurados[0]["placa_real"] == "SEM_VEICULO"
    assert regs_restaurados[0]["modelo_real"] == "SEM_VEICULO"
    assert regs_restaurados[0]["cor_real"] == "-"

    # Relatório QA contabiliza como Falso positivo
    relatorio = conferir.gerar_relatorio_qa(xlsx, caminho_gabarito=gabarito_path)
    assert "Falso positivo" in relatorio
    assert "| dia_frontal | 1 | 0 | 0 | 1 |" in relatorio
