"""
Testes automatizados do validador sintático, limpeza, máscara e desempate de placas brasileiras.
"""

import pytest

from src.validator import (
    desempatar_caracteres,
    desempatar_placas,
    limpar_texto,
    obter_placa_ou_mascara,
    validar_placa,
)


# ---------------------------------------------------------------------------
# 1. limpar_texto
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entrada, esperado",
    [
        ("", ""),
        (None, ""),
        ("   ", ""),
        ("abc-1234", "ABC1234"),
        ("  xyz 9z99  ", "XYZ9Z99"),
        ("BRASIL ABC1D23", "ABC1D23"),
        ("MERCOSUL ABC-1D23", "ABC1D23"),
        ("BR ABC1234", "ABC1234"),
        ("BR-BRASIL-MERCOSUL", ""),
        ("|ABC.1D/23*", "ABC1D23"),
        ("BRA-ABC1D23", "BRAABC1D23"),
    ],
)
def test_limpar_texto(entrada, esperado):
    assert limpar_texto(entrada) == esperado


# ---------------------------------------------------------------------------
# 2. validar_placa - Casos válidos (Mercosul, Antiga, Correções, Prioridade)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entrada, esperado",
    [
        # Placas Mercosul válidas
        ("ABC1D23", "ABC1D23"),
        ("ELU9D76", "ELU9D76"),
        ("BRA2E19", "BRA2E19"),
        ("NXZ9F22", "NXZ9F22"),
        ("abc1d23", "ABC1D23"),
        ("XYZ-9Z99", "XYZ9Z99"),
        ("BRASIL ABC1D23", "ABC1D23"),
        ("MERCOSUL ELU9D76", "ELU9D76"),
        ("BRA-ABC1D23", "ABC1D23"),
        ("|ABC1D23", "ABC1D23"),
        # Placas antigas válidas
        ("ABC1234", "ABC1234"),
        ("abc-1234", "ABC1234"),
        ("XYZ9999", "XYZ9999"),
        ("KLM1523", "KLM1523"),
        ("BR ABC-1234", "ABC1234"),
        # Prioridade Mercosul na 5ª posição (índice 4: '0' -> 'D', '8' -> 'B')
        ("ELU9076", "ELU9D76"),
        ("ABC1823", "ABC1B23"),
        ("BRA-ELU9076", "ELU9D76"),
        ("BRASIL ABC1823", "ABC1B23"),
        # Correção de dígitos em posições de letra (0, 1, 2)
        ("4BC1234", "ABC1234"),
        ("AB01234", "ABO1234"),
        ("1BC1234", "IBC1234"),
        ("2BC1234", "ZBC1234"),
        ("5BC1234", "SBC1234"),
        ("6BC1234", "GBC1234"),
        ("7BC1234", "TBC1234"),
        ("8BC1234", "BBC1234"),
        # Correção de letras em posições de dígito (3, 5, 6)
        ("ABCO234", "ABC0234"),
        ("ABC123O", "ABC1230"),
        ("ABC1D2O", "ABC1D20"),
        ("ABC12ZS", "ABC1225"),
        ("4BC1D2O", "ABC1D20"),
        ("ABCIB34", "ABC1B34"),
    ],
)
def test_validar_placa_validas_e_corrigidas(entrada, esperado):
    assert validar_placa(entrada) == esperado


# ---------------------------------------------------------------------------
# 3. validar_placa - Entradas inválidas / rejeitadas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entrada",
    [
        "",
        None,
        "   ---   ",
        "ABC123",          # tamanho < 7
        "BR-ABC123",       # tamanho < 7 após remoção do cabeçalho BR
        "1234567",         # '3' não mapeia para letra na pos 2
        "A3C1234",         # '3' sem substituição na pos 1
        "ABC123C",         # 'C' sem substituição na pos 6
        "9999999",         # '9' não mapeia para letra
        "4801234",         # exige 3 correções (custo > 2)
    ],
)
def test_validar_placa_rejeita_invalidas(entrada):
    assert validar_placa(entrada) is None


# ---------------------------------------------------------------------------
# 4. obter_placa_ou_mascara
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entrada, esperado",
    [
        (None, "???????"),
        ("", "???????"),
        ("---", "???????"),
        ("ABC1D23", "ABC1D23"),
        ("abc-1234", "ABC1234"),
        ("ELU9076", "ELU9D76"),
        ("S4", "S??4???"),
        ("ABC", "ABC????"),
        ("1234", "???1234"),
        ("AB?1234", "AB?1234"),
        ("ABC1?23", "ABC123?"),
        ("A3C1234", "A?C1234"),
        ("A3C12345", "A?C1234"),
    ],
)
def test_obter_placa_ou_mascara(entrada, esperado):
    assert obter_placa_ou_mascara(entrada) == esperado


# ---------------------------------------------------------------------------
# 5. desempatar_caracteres e desempatar_placas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "char1, char2, pos, esperado",
    [
        ("L", "Z", 1, "L"),
        ("Z", "L", 1, "L"),
        ("D", "0", 4, "D"),
        ("0", "D", 4, "D"),
        ("B", "8", 0, "B"),
        ("8", "B", 4, "B"),
        ("I", "1", 2, "I"),
        ("O", "0", 1, "O"),
        ("0", "D", 3, "0"),
        ("8", "B", 5, "8"),
        ("1", "I", 6, "1"),
        ("A", "A", 0, "A"),
    ],
)
def test_desempatar_caracteres(char1, char2, pos, esperado):
    assert desempatar_caracteres(char1, char2, pos) == esperado


@pytest.mark.parametrize(
    "placa1, placa2, esperado",
    [
        ("EZU9D76", "ELU9076", "ELU9D76"),
        ("ABC1823", "ABC1B23", "ABC1B23"),
        ("ABCI234", "ABC1234", "ABC1234"),
        ("ZBC1D23", "LBC1023", "LBC1D23"),
        ("", "ABC1D23", "ABC1D23"),
        ("ABC1D23", "", "ABC1D23"),
        ("ABC", "ABC1D23", "ABC"),
    ],
)
def test_desempatar_placas(placa1, placa2, esperado):
    assert desempatar_placas(placa1, placa2) == esperado
