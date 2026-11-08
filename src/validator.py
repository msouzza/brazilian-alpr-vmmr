"""
Valida se um texto lido pelo OCR corresponde a um formato de placa brasileira
(antiga ou Mercosul). Aplica limpeza, correção sintática inteligente e desempate
posicional de caracteres baseado no padrão Mercosul e histórico de confusões de OCR.
"""

import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Mapeamentos de confusões clássicas de OCR em placas veiculares
DIGITO_PARA_LETRA = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

# No padrão Mercosul (LLLNLNN), o 5º caractere (índice 4) é obrigatoriamente letra.
# A confusão clássica mais comum é '0' lido no lugar de 'D' (fonte de placa e ruído de resolução)
# e '8' lido no lugar de 'B'.
DIGITO_PARA_LETRA_POS4 = {
    "0": "D",  # Prioridade absoluta para Mercosul: '0' -> 'D'
    "8": "B",  # '8' -> 'B'
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
}

LETRA_PARA_DIGITO = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "b": "6",
    "T": "7",
    "B": "8",
    "g": "9",
    "q": "9",
}

# Preferências em caso de empate entre caracteres em posições de letra
# (char_preferido, char_desfavorecido)
PREFERENCIAS_DESEMPATE_LETRAS = {
    ("L", "Z"),  # Em placas veiculares, 'L' frequentemente tem a base confundida com 'Z' pelo OCR
    ("D", "0"),  # 'D' frequentemente vira '0' no 5º caractere
    ("B", "8"),  # 'B' frequentemente vira '8'
    ("I", "1"),
    ("O", "0"),
}

# Preferências em caso de empate entre caracteres em posições de dígito
PREFERENCIAS_DESEMPATE_DIGITOS = {
    ("0", "O"),
    ("0", "D"),
    ("8", "B"),
    ("1", "I"),
    ("1", "L"),
    ("2", "Z"),
}


def limpar_texto(texto: str) -> str:
    """Remove espaços, traços, pontuação e palavras comuns de borda (ex: BR, BRASIL)."""
    if not texto:
        return ""
    texto = texto.upper().strip()
    # Remove palavras frequentes de cabeçalho de placa Mercosul se coladas
    texto = re.sub(r"\b(BRASIL|MERCOSUL|BR)\b", "", texto)
    # Remove qualquer caracter que não seja alfanumérico
    texto = re.sub(r"[^A-Z0-9]", "", texto)
    return texto


def _avaliar_candidato_mercosul(cand: str) -> tuple[str | None, int]:
    """
    Retorna (placa_corrigida, num_alteracoes) para o padrão Mercosul (LLLNLNN).
    Se inválido, retorna (None, 999).
    """
    if len(cand) != 7:
        return None, 999

    chars = list(cand)
    alteracoes = 0

    # Índices 0, 1, 2 devem ser letras
    for i in (0, 1, 2):
        if chars[i].isdigit():
            subst = DIGITO_PARA_LETRA.get(chars[i])
            if subst:
                chars[i] = subst
                alteracoes += 1
            else:
                return None, 999
        elif not chars[i].isalpha():
            return None, 999

    # Índice 3 deve ser dígito
    if chars[3].isalpha():
        subst = LETRA_PARA_DIGITO.get(chars[3])
        if subst:
            chars[3] = subst
            alteracoes += 1
        else:
            return None, 999
    elif not chars[3].isdigit():
        return None, 999

    # Índice 4 deve ser letra (usa DIGITO_PARA_LETRA_POS4 onde '0' -> 'D' e '8' -> 'B')
    if chars[4].isdigit():
        subst = DIGITO_PARA_LETRA_POS4.get(chars[4])
        if subst:
            chars[4] = subst
            alteracoes += 1
        else:
            return None, 999
    elif not chars[4].isalpha():
        return None, 999

    # Índices 5, 6 devem ser dígitos
    for i in (5, 6):
        if chars[i].isalpha():
            subst = LETRA_PARA_DIGITO.get(chars[i])
            if subst:
                chars[i] = subst
                alteracoes += 1
            else:
                return None, 999
        elif not chars[i].isdigit():
            return None, 999

    resultado = "".join(chars)
    if re.match(config.REGEX_PLACA_MERCOSUL, resultado):
        return resultado, alteracoes
    return None, 999


def _avaliar_candidato_antiga(cand: str) -> tuple[str | None, int]:
    """
    Retorna (placa_corrigida, num_alteracoes) para o padrão Antigo (LLLNNNN).
    Se inválido, retorna (None, 999).
    """
    if len(cand) != 7:
        return None, 999

    chars = list(cand)
    alteracoes = 0

    # Índices 0, 1, 2 devem ser letras
    for i in (0, 1, 2):
        if chars[i].isdigit():
            subst = DIGITO_PARA_LETRA.get(chars[i])
            if subst:
                chars[i] = subst
                alteracoes += 1
            else:
                return None, 999
        elif not chars[i].isalpha():
            return None, 999

    # Índices 3, 4, 5, 6 devem ser dígitos
    for i in (3, 4, 5, 6):
        if chars[i].isalpha():
            subst = LETRA_PARA_DIGITO.get(chars[i])
            if subst:
                chars[i] = subst
                alteracoes += 1
            else:
                return None, 999
        elif not chars[i].isdigit():
            return None, 999

    resultado = "".join(chars)
    if re.match(config.REGEX_PLACA_ANTIGA, resultado):
        return resultado, alteracoes
    return None, 999


def validar_placa(texto: str) -> str | None:
    """
    Retorna a placa formatada se for válida (direta ou corrigida sintaticamente),
    ou None se não bater com nenhum dos dois formatos (antigo ou Mercosul).
    Usa princípio de mínima alteração para desambiguação, priorizando o padrão
    Mercosul em casos de dúvida ou confusão clássica no 5º caractere ('0' -> 'D', '8' -> 'B').
    """
    texto_limpo = limpar_texto(texto)
    if not texto_limpo or len(texto_limpo) < 7:
        return None

    # 1. Teste direto exato se tamanho for 7
    if len(texto_limpo) == 7:
        # Se for diretamente Mercosul válido (LLLNLNN), aceita imediatamente
        if re.match(config.REGEX_PLACA_MERCOSUL, texto_limpo):
            return texto_limpo

        # Se casar com placa antiga (LLLNNNN), verifica ambiguidade no 5º caractere (índice 4):
        # Se o 5º caractere for '0' (confusão com 'D') ou '8' (confusão com 'B'), prioriza Mercosul!
        if re.match(config.REGEX_PLACA_ANTIGA, texto_limpo):
            if texto_limpo[4] == "0":
                cand_mercosul = texto_limpo[:4] + "D" + texto_limpo[5:]
                if re.match(config.REGEX_PLACA_MERCOSUL, cand_mercosul):
                    return cand_mercosul
            elif texto_limpo[4] == "8":
                cand_mercosul = texto_limpo[:4] + "B" + texto_limpo[5:]
                if re.match(config.REGEX_PLACA_MERCOSUL, cand_mercosul):
                    return cand_mercosul
            return texto_limpo

    # 2. Gera janelas de 7 caracteres se houver prefixos/sufixos ou ruídos de borda
    # Ex: "BRAABC1234" -> testa "BRAABC1", "RAABC12", "AABC123", "ABC1234"
    candidatos = []
    for i in range(len(texto_limpo) - 6):
        candidatos.append(texto_limpo[i : i + 7])

    melhor_placa = None
    menor_custo = 999

    for cand in candidatos:
        # Se for diretamente válida Mercosul na janela, prioridade máxima
        if re.match(config.REGEX_PLACA_MERCOSUL, cand):
            return cand

        # Se for diretamente válida Antiga, verifica se pode ser Mercosul com '0' -> 'D' ou '8' -> 'B'
        if re.match(config.REGEX_PLACA_ANTIGA, cand):
            if cand[4] == "0":
                return cand[:4] + "D" + cand[5:]
            if cand[4] == "8":
                return cand[:4] + "B" + cand[5:]
            return cand

        # Avalia correções possíveis
        placa_m, custo_m = _avaliar_candidato_mercosul(cand)
        placa_a, custo_a = _avaliar_candidato_antiga(cand)

        # Desempate com prioridade Mercosul:
        # Se ambos os padrões forem possíveis:
        # - Se custo_m <= custo_a: escolhe Mercosul
        # - Se custo_m == custo_a + 1 e a diferença for a substituição de '0' por 'D' no 5º caractere: prioriza Mercosul!
        cand_venc = None
        custo_venc = 999

        if placa_m and placa_a:
            if custo_m <= custo_a:
                cand_venc = placa_m
                custo_venc = custo_m
            elif custo_m == custo_a + 1 and cand[4] in ("0", "8"):
                cand_venc = placa_m
                custo_venc = custo_a
            else:
                cand_venc = placa_a
                custo_venc = custo_a
        elif placa_m:
            cand_venc = placa_m
            custo_venc = custo_m
        elif placa_a:
            cand_venc = placa_a
            custo_venc = custo_a

        if cand_venc and custo_venc < menor_custo:
            menor_custo = custo_venc
            melhor_placa = cand_venc

    # Aceita se exigiu no máximo 2 correções de confusão de OCR
    if melhor_placa and menor_custo <= 2:
        return melhor_placa

    return None


def desempatar_caracteres(char1: str, char2: str, posicao: int) -> str:
    """
    Desempata dois caracteres concorrentes na mesma posição baseado nas regras
    sintáticas da placa brasileira e histórico de confusões de OCR (ex: Z vs L, 0 vs D, 8 vs B).
    """
    if char1 == char2:
        return char1

    c1, c2 = char1.upper(), char2.upper()

    # Posições 0, 1, 2 e 4 (quando Mercosul) esperam LETRAS
    eh_posicao_letra = (posicao in (0, 1, 2, 4))
    if eh_posicao_letra:
        if (c1, c2) in PREFERENCIAS_DESEMPATE_LETRAS:
            return c1
        if (c2, c1) in PREFERENCIAS_DESEMPATE_LETRAS:
            return c2
        # Se um for letra e outro for dígito, prefere letra
        if c1.isalpha() and c2.isdigit():
            return c1
        if c2.isalpha() and c1.isdigit():
            return c2
    else:
        # Posições 3, 5, 6 esperam DÍGITOS
        if (c1, c2) in PREFERENCIAS_DESEMPATE_DIGITOS:
            return c1
        if (c2, c1) in PREFERENCIAS_DESEMPATE_DIGITOS:
            return c2
        if c1.isdigit() and c2.isalpha():
            return c1
        if c2.isdigit() and c1.isalpha():
            return c2

    return char1


def desempatar_placas(placa1: str, placa2: str) -> str:
    """
    Reconcilia e desempata duas placas concorrentes posição a posição,
    resolvendo ambiguidades clássicas (Z vs L, 0 vs D, 8 vs B).
    """
    if not placa1:
        return placa2 or ""
    if not placa2:
        return placa1

    if len(placa1) != 7 or len(placa2) != 7:
        return placa1

    chars_res = []
    for pos, (c1, c2) in enumerate(zip(placa1, placa2)):
        c_venc = desempatar_caracteres(c1, c2, pos)
        chars_res.append(c_venc)

    res = "".join(chars_res)
    val = validar_placa(res)
    return val if val else res


def obter_placa_ou_mascara(texto: str | None) -> str:
    """
    Retorna a placa formatada se for 100% válida.
    Caso contrário, preenche os caracteres desconhecidos com '?' mantendo
    a estrutura padrão brasileira de 7 caracteres (ex: 'ABC?D23', 'S4?????', '???????').
    """
    if not texto:
        return "???????"

    # 1. Tenta validação completa primeiro
    placa_valida = validar_placa(texto)
    if placa_valida:
        return placa_valida

    texto_limpo = limpar_texto(texto)
    if not texto_limpo:
        return "???????"

    # Se tiver exatamente 7 caracteres, analisa posição a posição
    if len(texto_limpo) == 7:
        chars = list(texto_limpo)
        for i in range(7):
            if i in (0, 1, 2):
                if chars[i].isdigit():
                    chars[i] = DIGITO_PARA_LETRA.get(chars[i], "?")
                elif not chars[i].isalpha():
                    chars[i] = "?"
            elif i == 3:
                if chars[i].isalpha():
                    chars[i] = LETRA_PARA_DIGITO.get(chars[i], "?")
                elif not chars[i].isdigit():
                    chars[i] = "?"
            elif i == 4:
                # Pode ser letra (Mercosul) ou dígito (Antiga)
                if not (chars[i].isalnum()):
                    chars[i] = "?"
            elif i in (5, 6):
                if chars[i].isalpha():
                    chars[i] = LETRA_PARA_DIGITO.get(chars[i], "?")
                elif not chars[i].isdigit():
                    chars[i] = "?"
        return "".join(chars)

    # Se tiver mais de 7 caracteres, pega os primeiros 7 ou trunca
    if len(texto_limpo) > 7:
        return obter_placa_ou_mascara(texto_limpo[:7])

    # Se tiver menos de 7 caracteres, alinha e preenche o restante com '?'
    # Estrutura base: 3 letras + 1 dígito + 1 letra/dígito + 2 dígitos
    letras = [c for c in texto_limpo if c.isalpha()]
    digitos = [c for c in texto_limpo if c.isdigit()]

    resultado = ["?"] * 7
    # Preenche até 3 letras no início
    for i, l in enumerate(letras[:3]):
        resultado[i] = l

    # Preenche dígitos restantes a partir da posição 3
    pos_d = 3
    for d in digitos:
        while pos_d < 7 and resultado[pos_d] != "?":
            pos_d += 1
        if pos_d < 7:
            resultado[pos_d] = d

    # Se o texto lido não conseguiu ser distribuído, posiciona na ordem direta
    mascara = "".join(resultado)
    if mascara == "???????":
        chars = list(texto_limpo[:7])
        while len(chars) < 7:
            chars.append("?")
        return "".join(chars)

    return mascara


if __name__ == "__main__":
    testes = [
        "abc1234",          # minúsculo
        "ABC-1234",         # traço
        "abc1d23",          # Mercosul minúsculo
        "XYZ 9Z99",         # Mercosul com espaço
        "4BC1234",          # '4' no lugar do 'A' -> ABC1234
        "AB01234",          # '0' no lugar do 'O' -> ABO1234
        "ABCIB34",          # 'I' e 'B' no lugar de '1' e '8' -> ABC1834
        "ABC1B23",          # Mercosul válido direto
        "ABC1823",          # Ambiguidade: pode ser antiga com 8 na pos 4 -> ABC1823
        "BRA-ABC1D23",      # com prefixo BRA
        "|ABC1D23",         # com ruído de borda
    ]
    print("--- Testes de Validação e Correção Sintática ---")
    for t in testes:
        resultado = validar_placa(t)
        print(f"{t!r:18} -> {resultado}")

    print("\n--- Testes de Máscara com '?' ---")
    testes_mascara = ["", "S4", "ABC", "1234", "AB?1234", "ABC1?23", "X1", None]
    for t in testes_mascara:
        print(f"{str(t)!r:18} -> {obter_placa_ou_mascara(t)}")
