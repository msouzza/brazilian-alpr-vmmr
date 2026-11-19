"""
Testes automatizados para o script avaliar_alpr.py usando planilhas sintéticas temporárias.
"""

from openpyxl import Workbook

import avaliar_alpr
from src.spreadsheet_writer import COLUNAS


def _criar_planilha_sintetica(caminho, linhas_dados):
    wb = Workbook()
    ws = wb.active
    ws.title = "Placas"
    cabecalho = list(COLUNAS) + ["Placa real", "Modelo real", "Condição"]
    ws.append(cabecalho)
    for row in linhas_dados:
        ws.append(row)
    wb.save(caminho)


def _linha_base(
    placa,
    d1,
    d2,
    d3,
    conf_vmmr,
    placa_real,
    modelo_real,
    condicao,
):
    return [
        placa,                # Placa
        "Carro",              # Tipo
        "Prata / Cinza",      # Cor
        "Volkswagen Fox",     # Marca / Modelo
        d1,                   # Descoberta 1 (%)
        d2,                   # Descoberta 2 (%)
        d3,                   # Descoberta 3 (%)
        conf_vmmr,            # Confiabilidade VMMR
        "18/11/2026",         # Data
        "14:00:00",           # Hora
        0.90,                 # Confiança OCR
        "data/fotos/f.jpg",   # Foto
        "video.mp4",          # Vídeo de origem
        placa_real,           # Placa real
        modelo_real,          # Modelo real
        condicao,             # Condição
    ]


def test_avaliar_alpr_trava_amostra_pequena(tmp_path, capsys):
    xlsx = tmp_path / "amostra_pequena.xlsx"
    linhas = [
        _linha_base(
            "ABC1D23",
            "1º Volkswagen Fox (73.5%)",
            "2º Hyundai HB20 (12.0%)",
            "3º Fiat Argo (5.0%)",
            "Alta",
            "ABC1D23",
            "volkswagen fox",
            "Dia",
        ),
        _linha_base(
            "ELU9D76",
            "1º Fiat Palio (45.0%)",
            "2º Chevrolet Onix (35.0%)",
            "3º Citroën C3 (10.0%)",
            "Média",
            "ELU9D75",  # Hamming == 1
            "citroen c3",  # Top 3, sem acento e minúscula
            "Dia",
        ),
        _linha_base(
            "SEM_PLACA",
            "1º Toyota Corolla (30.0%)",
            "2º Honda Civic (28.0%)",
            "3º Nissan Sentra (15.0%)",
            "Baixa / Ambígua",
            "SEM_PLACA",
            "Honda Civic",
            "Dia",
        ),
    ]
    _criar_planilha_sintetica(xlsx, linhas)

    saida_md = tmp_path / "resultados.md"
    codigo = avaliar_alpr.main([str(xlsx), "--saida", str(saida_md)])
    assert codigo == 0

    stdout = capsys.readouterr().out
    assert "amostra pequena (n < 15)" in stdout
    assert "%" not in stdout.split("## Métricas por Condição")[1]
    assert saida_md.exists()
    conteudo_md = saida_md.read_text(encoding="utf-8")
    assert "1/2 — amostra pequena (n < 15)" in conteudo_md  # placa exata: 1 de 2 legíveis
    assert "2/2 — amostra pequena (n < 15)" in conteudo_md  # <= 1 erro: 2 de 2 legíveis
    assert "1/3 — amostra pequena (n < 15)" in conteudo_md  # Top 1: 1 de 3
    assert "3/3 — amostra pequena (n < 15)" in conteudo_md  # Top 3: 3 de 3


def test_avaliar_alpr_metricas_completas_com_15_ou_mais_amostras(tmp_path):
    xlsx = tmp_path / "amostra_completa.xlsx"
    linhas = []

    # 16 veículos na condição "Dia":
    # - 12 com placa exata + modelo Top 1 exato ("Alta")
    # - 2 com 1 erro de Hamming na placa + modelo no Top 3 ("Média")
    # - 1 com erro > 1 na placa + modelo errado ("Baixa / Ambígua")
    # - 1 sem placa legível ("-") + modelo Top 1 certo ("Alta")
    for _ in range(12):
        linhas.append(
            _linha_base(
                "ABC1D23",
                "1º Volkswagen Fox (73.5%)",
                "2º Fiat Uno (10.0%)",
                "3º Fiat Palio (5.0%)",
                "Alta",
                "ABC-1D23",
                "Volkswagen Fox",
                "Dia",
            )
        )
    for _ in range(2):
        linhas.append(
            _linha_base(
                "ELU9D76",
                "1º Fiat Palio (41.0%)",
                "2º Citroën C3 (30.0%)",
                "3º Hyundai HB20 (12.0%)",
                "Média",
                "ELU9076",  # 1 char diferente (D vs 0)
                "citroen c3",  # ignora acento/maiúsculas, está no Top 3
                "Dia",
            )
        )
    linhas.append(
        _linha_base(
            "XYZ9999",
            "1º Chevrolet Onix (25.0%)",
            "2º Hyundai HB20 (24.0%)",
            "3º Ford Ka (20.0%)",
            "Baixa / Ambígua",
            "ABC1234",  # 7 chars diferentes
            "Toyota Corolla",  # fora do Top 3
            "Dia",
        )
    )
    linhas.append(
        _linha_base(
            "SEM_PLACA",
            "1º Toyota Hilux (82.0%)",
            "2º Chevrolet S10 (8.0%)",
            "3º Ford Ranger (3.0%)",
            "Alta",
            "-",
            "toyota hilux",
            "Dia",
        )
    )

    # 2 veículos em "Noite" (n < 15 -> bloqueia percentual apenas nesta linha)
    linhas.append(
        _linha_base(
            "BRA2E19",
            "1º Hyundai HB20 (60.0%)",
            "2º Volkswagen Gol (20.0%)",
            "3º Fiat Argo (10.0%)",
            "Alta",
            "BRA2E19",
            "Hyundai HB20",
            "Noite",
        )
    )
    linhas.append(
        _linha_base(
            "KLM1234",
            "1º Volkswagen Gol (50.0%)",
            "2º Renault Sandero (20.0%)",
            "3º Fiat Uno (15.0%)",
            "Alta",
            "KLM1234",
            "Fiat Uno",
            "Noite",
        )
    )

    _criar_planilha_sintetica(xlsx, linhas)
    registros = avaliar_alpr.carregar_registros(xlsx)
    relatorio = avaliar_alpr.gerar_relatorio_markdown(registros)

    # Condição Dia: 16 conferidos, 1 sem placa, 0 ilegível, 0 falso positivo -> 15 placas válidas, 16 modelos válidos, 0 cores válidas
    # Placa exata: 12/15 (80.0%) | <= 1 erro: 14/15 (93.3%)
    # Modelo Top 1: 13/16 (81.2%) | Modelo Top 3: 15/16 (93.8%)
    assert "| Dia | 16 | 1 | 0 | 0 | 12/15 (80.0%) | 14/15 (93.3%) | 13/16 (81.2%) | 15/16 (93.8%) | 0/0 — amostra pequena (n < 15) |" in relatorio

    # Condição Noite: n=2 < 15 -> sem percentual
    assert "| Noite | 2 | 0 | 0 | 0 | 2/2 — amostra pequena (n < 15) | 2/2 — amostra pequena (n < 15) | 1/2 — amostra pequena (n < 15) | 2/2 — amostra pequena (n < 15) | 0/0 — amostra pequena (n < 15) |" in relatorio

    # Calibração Alta: 12 + 1 + 2 = 15 amostras -> exibe percentual: 14/15 (93.3%) Top 1, 15/15 (100.0%) Top 3
    assert "| Alta | 15 | 14/15 (93.3%) | 15/15 (100.0%) |" in relatorio
    # Calibração Média: 2 amostras < 15
    assert "| Média | 2 | 0/2 — amostra pequena (n < 15) | 2/2 — amostra pequena (n < 15) |" in relatorio


def test_avaliar_alpr_sem_placa_vs_ilegivel_parcial_e_cor_real(tmp_path):
    xlsx = tmp_path / "com_cor_e_ilegivel.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Placas"
    ws.append(list(COLUNAS) + ["Placa real", "Modelo real", "Cor real", "Condição"])

    def _row(placa_pred, cor_pred, placa_real, cor_real, cond="Dia"):
        return [
            placa_pred,
            "Carro",
            cor_pred,
            "Volkswagen Fox",
            "1º Volkswagen Fox (73.5%)",
            "2º Hyundai HB20 (12.0%)",
            "3º Fiat Argo (5.0%)",
            "Alta",
            "18/11/2026",
            "14:00:00",
            0.90,
            "data/fotos/f.jpg",
            "video.mp4",
            placa_real,
            "Volkswagen Fox",
            cor_real,
            cond,
        ]

    # 12 exatas + cor certa (inclui variação sem acento/minúscula)
    for i in range(12):
        cor_p = "Amarelo / Laranja" if i == 0 else "Prata / Cinza"
        cor_r = "amarelo / laranja" if i == 0 else "prata / cinza"
        ws.append(_row("ABC1D23", cor_p, "ABC1D23", cor_r))

    # 2 parcialmente legíveis com $ (GWB$$$$):
    # - 1 exata nas posições legíveis (GWB8R01 vs GWB$$$$) + cor errada
    # - 1 com 1 erro Hamming nas posições legíveis (GWC8R01 vs GWB$$$$) + cor certa
    ws.append(_row("GWB8R01", "Branco", "GWB$$$$", "Preto"))
    ws.append(_row("GWC8R01", "Roxo / Vinho", "GWB$$$$", "roxo / vinho"))

    # 1 erro > 1 nas posições legíveis (XYZ8R01 vs GWB$$$$) + cor certa
    ws.append(_row("XYZ8R01", "Azul", "GWB$$$$", "azul"))

    # 2 SEM_PLACA (1 com cor certa, 1 sem cor preenchida)
    ws.append(_row("SEM_PLACA", "Verde", "SEM_PLACA", "Verde"))
    ws.append(_row("SEM_PLACA", "Preto", "-", ""))

    # 2 Ilegíveis (1 com '$$$$$$$', 1 com 'ILEGIVEL') - não devem entrar no denominador de placas!
    ws.append(_row("BRA2E19", "Branco", "$$$$$$$", "branco"))
    ws.append(_row("KLM1234", "Vermelho", "ILEGIVEL", "Preto"))

    wb.save(xlsx)

    assert avaliar_alpr.eh_placa_ilegivel("$$$$$$$") is True
    assert avaliar_alpr.eh_placa_ilegivel("ILEGIVEL") is True
    assert avaliar_alpr.eh_placa_ilegivel("ilegível") is True
    assert avaliar_alpr.eh_placa_ilegivel("GWB$$$$") is False
    assert avaliar_alpr.eh_placa_ilegivel("SEM_PLACA") is False

    registros = avaliar_alpr.carregar_registros(xlsx)
    assert len(registros) == 19

    # Parciais GWB$$$$
    assert registros[12].placa_exata is True
    assert registros[12].placa_ate_1_erro is True
    assert registros[12].cor_correta is False
    assert registros[13].placa_exata is False
    assert registros[13].placa_ate_1_erro is True
    assert registros[13].cor_correta is True
    assert registros[14].placa_exata is False
    assert registros[14].placa_ate_1_erro is False

    # Sem placa vs Ilegível
    assert registros[15].sem_placa is True
    assert registros[15].placa_ilegivel is False
    assert registros[15].sem_placa_legivel is True
    assert registros[17].sem_placa is False
    assert registros[17].placa_ilegivel is True
    assert registros[17].sem_placa_legivel is True
    assert registros[17].placa_exata is False
    assert registros[17].placa_ate_1_erro is False
    assert registros[18].placa_ilegivel is True
    assert registros[18].placa_exata is False

    relatorio = avaliar_alpr.gerar_relatorio_markdown(registros)
    # 19 conferidos: 2 sem placa, 2 ilegíveis -> 15 placas válidas:
    # Placa exata: 12 + 1 = 13/15 (86.7%)
    # Placa <= 1 erro: 12 + 1 + 1 = 14/15 (93.3%)
    # Cor válida: 18 preenchidas (1 linha com cor vazia '-') -> 12 + 1 + 1 + 1 + 1 = 16/18 (88.9%)
    assert (
        "| Dia | 19 | 2 | 2 | 0 | 13/15 (86.7%) | 14/15 (93.3%) | 19/19 (100.0%) | 19/19 (100.0%) | 16/18 (88.9%) |"
        in relatorio
    )
    assert (
        "| **Total Geral** | **19** | **2** | **2** | **0** | **13/15 (86.7%)** | **14/15 (93.3%)** | **19/19 (100.0%)** | **19/19 (100.0%)** | **16/18 (88.9%)** |"
        in relatorio
    )


def test_avaliar_alpr_falso_positivo_fora_dos_denominadores(tmp_path):
    assert avaliar_alpr.VALORES_FALSO_POSITIVO == {
        "SEM_VEICULO",
        "SEM VEICULO",
        "FALSO_POSITIVO",
        "FALSO POSITIVO",
    }
    assert avaliar_alpr.eh_falso_positivo("SEM_VEICULO") is True
    assert avaliar_alpr.eh_falso_positivo("sem veiculo") is True
    assert avaliar_alpr.eh_falso_positivo("falso_positivo") is True
    assert avaliar_alpr.eh_falso_positivo("Falso Positivo") is True
    assert avaliar_alpr.eh_falso_positivo("ABC1D23", "SEM_VEICULO") is True
    assert avaliar_alpr.eh_falso_positivo("SEM_PLACA", "Volkswagen Fox") is False
    assert avaliar_alpr.eh_falso_positivo(None, None) is False

    xlsx = tmp_path / "falso_positivo.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Placas"
    ws.append(list(COLUNAS) + ["Placa real", "Modelo real", "Cor real", "Condição"])

    def _row(placa_pred, cor_pred, placa_real, modelo_real, cor_real, conf="Alta", cond="Dia"):
        return [
            placa_pred,
            "Carro",
            cor_pred,
            "Volkswagen Fox",
            "1º Volkswagen Fox (73.5%)",
            "2º Hyundai HB20 (12.0%)",
            "3º Fiat Argo (5.0%)",
            conf,
            "18/11/2026",
            "14:00:00",
            0.90,
            "data/fotos/f.jpg",
            "video.mp4",
            placa_real,
            modelo_real,
            cor_real,
            cond,
        ]

    # 15 veículos reais com 100% de acerto
    for _ in range(15):
        ws.append(_row("ABC1D23", "Prata / Cinza", "ABC1D23", "Volkswagen Fox", "Prata / Cinza"))

    # 3 falsos positivos com variações e confiabilidades distintas (inclusive simulando predição preenchida)
    ws.append(_row("SEM_PLACA", "Prata / Cinza", "SEM_VEICULO", "SEM_VEICULO", "-", conf="Alta"))
    ws.append(_row("ABC1D23", "Branco", "FALSO_POSITIVO", "Volkswagen Fox", "Branco", conf="Média"))
    ws.append(_row("SEM_PLACA", "Preto", "SEM_PLACA", "sem veiculo", "-", conf="Baixa / Ambígua"))

    wb.save(xlsx)

    registros = avaliar_alpr.carregar_registros(xlsx)
    assert len(registros) == 18

    for r in registros[15:]:
        assert r.falso_positivo is True
        assert r.sem_placa_legivel is False
        assert r.placa_ilegivel is False
        assert r.placa_exata is False
        assert r.placa_ate_1_erro is False
        assert r.modelo_top1_acerto is False
        assert r.modelo_top3_acerto is False
        assert r.cor_correta is False

    relatorio = avaliar_alpr.gerar_relatorio_markdown(registros)
    assert "| Condição | Conferidos | Sem placa | Ilegível | Falso positivo |" in relatorio
    assert (
        "| Dia | 18 | 0 | 0 | 3 | 15/15 (100.0%) | 15/15 (100.0%) | 15/15 (100.0%) | 15/15 (100.0%) | 15/15 (100.0%) |"
        in relatorio
    )
    assert (
        "| **Total Geral** | **18** | **0** | **0** | **3** | **15/15 (100.0%)** | **15/15 (100.0%)** | **15/15 (100.0%)** | **15/15 (100.0%)** | **15/15 (100.0%)** |"
        in relatorio
    )
    # Calibração VMMR não inclui os falsos positivos (nem Média/Baixa exclusivos de FP)
    assert "| Alta | 15 | 15/15 (100.0%) | 15/15 (100.0%) |" in relatorio
    assert "| Média |" not in relatorio
    assert "| Baixa / Ambígua |" not in relatorio
