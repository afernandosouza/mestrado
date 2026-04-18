import re

# Tabela de intervalos de codepoints por idioma
# Cada idioma pode ter múltiplos intervalos válidos
LANGUAGE_RANGES = {
    "pt":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "en":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "fr":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "it":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "es":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "de":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "ca":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "eo":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "fi":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "gl":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "nl":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "ro":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "cs":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "pl":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "hr":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "id":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "tr":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "az":  [(0x0009, 0x000D), (0x0020, 0x024F)],
    "ru":  [(0x0009, 0x000D), (0x0020, 0x007E), (0x0400, 0x04FF)], # ASCII + Cirílico
    "be":  [(0x0009, 0x000D), (0x0020, 0x007E), (0x0400, 0x04FF)],
    "bg":  [(0x0009, 0x000D), (0x0020, 0x007E), (0x0400, 0x04FF)],
    "he":  [(0x0009, 0x000D), (0x0020, 0x007E), (0x0590, 0x05FF)], # ASCII + Hebraico
    "ar":  [(0x0009, 0x000D), (0x0020, 0x007E), (0x0600, 0x06FF)], # ASCII + Árabe
    "fa":  [(0x0009, 0x000D), (0x0020, 0x007E), (0x0600, 0x06FF)],
    "ps":  [(0x0009, 0x000D), (0x0020, 0x007E), (0x0600, 0x06FF)],
    "ckb": [(0x0009, 0x000D), (0x0020, 0x007E), (0x0600, 0x06FF)],
    "arz": [(0x0009, 0x000D), (0x0020, 0x007E), (0x0600, 0x06FF)],
    "hi":  [(0x0009, 0x000D), (0x0020, 0x007E), (0x0900, 0x097F)], # ASCII + Devanagari
    "ta":  [(0x0009, 0x000D), (0x0020, 0x007E), (0x0B80, 0x0BFF)], # ASCII + Tâmil
}


def is_char_valid(char: str, ranges: list[tuple[int, int]]) -> bool:
    """Verifica se um caractere está dentro de algum dos intervalos válidos."""
    cp = ord(char)
    return any(min_cp <= cp <= max_cp for min_cp, max_cp in ranges)


def validate_string(text: str, language: str) -> dict:
    """
    Valida uma string para um dado idioma.

    Retorna um dicionário com:
    - valid: bool — True se todos os caracteres são válidos
    - invalid_chars: lista de caracteres inválidos com seus codepoints
    - coverage: float — percentual de caracteres válidos
    - language: idioma usado na validação
    """
    if language not in LANGUAGE_RANGES:
        raise ValueError(
            f"Idioma '{language}' não suportado. "
            f"Disponíveis: {sorted(LANGUAGE_RANGES.keys())}"
        )

    ranges = LANGUAGE_RANGES[language]
    invalid_chars = []

    for i, char in enumerate(text):
        if not is_char_valid(char, ranges):
            invalid_chars.append({
                "char": char,
                "codepoint": f"U+{ord(char):04X}",
                "position": i,
            })

    total = len(text)
    valid_count = total - len(invalid_chars)
    coverage = (valid_count / total * 100) if total > 0 else 100.0

    return {
        "language": language,
        "valid": len(invalid_chars) == 0,
        "total_chars": total,
        "invalid_count": len(invalid_chars),
        "invalid_chars": invalid_chars,
        "coverage": round(coverage, 2),
    }


def sanitize_string(
    text: str,
    language: str,
    replacement: str = "",
    strip_result: bool = True,
) -> dict:
    """
    Remove ou substitui os caracteres inválidos para o idioma informado.

    Parâmetros:
    - text: string de entrada
    - language: código do idioma (ex.: 'pt', 'ru', 'ar')
    - replacement: caractere usado no lugar dos inválidos (padrão: '' remove)
    - strip_result: se True, remove espaços duplos e espaços nas bordas

    Retorna um dicionário com:
    - original: texto original
    - sanitized: texto limpo
    - removed_chars: lista de caracteres removidos/substituídos
    - language: idioma usado
    """
    if language not in LANGUAGE_RANGES:
        raise ValueError(
            f"Idioma '{language}' não suportado. "
            f"Disponíveis: {sorted(LANGUAGE_RANGES.keys())}"
        )

    ranges = LANGUAGE_RANGES[language]
    sanitized_chars = []
    removed_chars = []

    for i, char in enumerate(text):
        if is_char_valid(char, ranges):
            sanitized_chars.append(char)
        else:
            removed_chars.append({
                "char": char,
                "codepoint": f"U+{ord(char):04X}",
                "position": i,
            })
            if replacement:
                sanitized_chars.append(replacement)

    sanitized = "".join(sanitized_chars)

    if strip_result:
        # Colapsa múltiplos espaços em um só e remove espaços nas bordas
        sanitized = re.sub(r" {2,}", " ", sanitized).strip()

    return {
        "language": language,
        "original": text,
        "sanitized": sanitized,
        "removed_count": len(removed_chars),
        "removed_chars": removed_chars,
    }


def sanitize_text(
    text: str,
    language: str
) -> str:
    if language not in LANGUAGE_RANGES:
        raise ValueError(
            f"Idioma '{language}' não suportado. "
            f"Disponíveis: {sorted(LANGUAGE_RANGES.keys())}"
        )

    ranges = LANGUAGE_RANGES[language]
    sanitized_chars = []
    removed_chars = []

    for i, char in enumerate(text):
        if is_char_valid(char, ranges):
            sanitized_chars.append(char)

    return "".join(sanitized_chars)


def validate_string_multi(text: str, languages: list[str]) -> dict[str, dict]:
    """
    Valida uma string contra múltiplos idiomas ao mesmo tempo.
    Útil para detectar qual idioma tem melhor cobertura.
    """
    results = {}
    for lang in languages:
        results[lang] = validate_string(text, lang)

    # Ordena por cobertura decrescente
    best = sorted(results.items(), key=lambda x: x[1]["coverage"], reverse=True)

    return {
        "results": results,
        "best_match": best[0][0] if best else None,
        "ranking": [lang for lang, _ in best],
    }


def print_report(result: dict) -> None:
    """Imprime um relatório legível da validação."""
    print(f"\n{'='*50}")
    print(f"  Idioma:       {result['language']}")
    print(f"  Válido:       {'✓ Sim' if result['valid'] else '✗ Não'}")
    print(f"  Total chars:  {result['total_chars']}")
    print(f"  Inválidos:    {result['invalid_count']}")
    print(f"  Cobertura:    {result['coverage']}%")

    if result["invalid_chars"]:
        print(f"\n  Caracteres fora do intervalo esperado:")
        for item in result["invalid_chars"]:
            print(
                f"    [{item['codepoint']}] '{item['char']}' "
                f"na posição {item['position']}"
            )
    print(f"{'='*50}")


def print_sanitize_report(result: dict) -> None:
    """Imprime um relatório legível da sanitização."""
    print(f"\n{'='*50}")
    print(f"  Idioma:        {result['language']}")
    print(f"  Original:      {result['original']}")
    print(f"  Sanitizado:    {result['sanitized']}")
    print(f"  Removidos:     {result['removed_count']}")

    if result["removed_chars"]:
        print(f"\n  Caracteres removidos:")
        for item in result["removed_chars"]:
            print(
                f"    [{item['codepoint']}] '{item['char']}' "
                f"na posição {item['position']}"
            )
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Exemplos de uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Exemplo 1: sanitizar português com emoji e cirílico ---
    texto_pt = "Olá, como você está? 😊 Привет"
    print("\n>>> Exemplo 1: Sanitizar português (remoção)")
    resultado = sanitize_string(texto_pt, "pt")
    print_sanitize_report(resultado)

    # --- Exemplo 2: sanitizar com substituição por placeholder ---
    print("\n>>> Exemplo 2: Sanitizar português (substituição por '?')")
    resultado = sanitize_string(texto_pt, "pt", replacement="?")
    print_sanitize_report(resultado)

    # --- Exemplo 3: sanitizar russo ---
    texto_ru = "Привет! Hello 😎 как дела?"
    print("\n>>> Exemplo 3: Sanitizar russo")
    resultado = sanitize_string(texto_ru, "ru")
    print_sanitize_report(resultado)

    # --- Exemplo 4: sanitizar árabe ---
    texto_ar = "مرحبا 🌍 كيف حالك hello"
    print("\n>>> Exemplo 4: Sanitizar árabe")
    resultado = sanitize_string(texto_ar, "ar")
    print_sanitize_report(resultado)

    # --- Exemplo 5: validação normal ainda funciona ---
    print("\n>>> Exemplo 5: Validação normal")
    resultado = validate_string(texto_pt, "pt")
    print_report(resultado)

    # --- Exemplo 6: multi-idioma ---
    texto_misto = "Hello, how are you?"
    print("\n>>> Exemplo 6: Multi-idioma")
    multi = validate_string_multi(texto_misto, ["en", "pt", "ru", "ar", "hi"])
    print(f"\n  Melhor match: {multi['best_match']}")
    print(f"  Ranking:      {multi['ranking']}")

    # --- Exemplo 8: retornar texto sanitizado ---
    texto_pt = "Olá, como você está? 😊 Привет"
    print("\n>>> Exemplo 1: Sanitizar português (remoção)")
    print("\n>>> texto original:", texto_pt)
    resultado = sanitize_text(texto_pt, "pt")
    print("\n>>> resultado:", resultado)