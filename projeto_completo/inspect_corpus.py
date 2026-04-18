# inspect_corpus.py
#
# Varre todos os textos do banco SQLite, detecta ocorrências de
# caracteres / padrões "estranhos" e gera um CSV com as colunas:
#   id | idioma | conteudo | tamanho | media_utf8
#
# Execução:
#   python inspect_corpus.py
# ------------------------------------------------------------------

import sys
import re
import csv
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------
# Ajuste de sys.path para encontrar config e demais módulos
# Assumindo que este arquivo está em projeto_completo/baseline_reproduction/
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import DATABASE, CHARS_TO_REMOVE, CODE_UTF8_TYPE, USAR_CONTEUDO_TRATADO
from signal_processing.text_signal import text_to_signal

# ------------------------------------------------------------------
# Configurações
# ------------------------------------------------------------------
DATABASE_PATH = ROOT_DIR / DATABASE
OUTPUT_CSV    = Path(__file__).resolve().parent / "corpus_anomalies.csv"

# ------------------------------------------------------------------
# Padrões que indicam conteúdo "estranho"
# Cada entrada é (nome_do_padrão, regex_compilado)
# ------------------------------------------------------------------
ANOMALY_PATTERNS = [
    # Tags HTML completas  <div>, </p>, <br />, etc.
    ("html_tag",            re.compile(r'<[a-zA-Z/][^>]{0,200}>', re.IGNORECASE)),

    # Entidades HTML  & &nbsp; &#160; &#x2F; etc.
    ("html_entity",         re.compile(r'&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]{2,8});')),

    # URLs / links
    ("url",                 re.compile(r'https?://\S+|www\.\S+')),

    # Sequências de escape  \n \t \r \uXXXX \xXX literais (como string)
    ("escape_sequence",     re.compile(r'\|$ntrxu][0-9a-fA-F]{0,4}')),

    # Caracteres de controle (U+0000–U+001F exceto tab/newline/CR)
    ("control_char",        re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')),

    # Caracteres substitutos / replacement (U+FFFD)
    ("replacement_char",    re.compile(r'\ufffd')),

    # Blocos de espaço em branco excessivo (4+ espaços seguidos, ou tabs)
    ("excessive_whitespace",re.compile(r'[ \t]{4,}')),

    # Emojis e símbolos miscelâneos (blocos Unicode U+1F300–U+1FAFF)
    ("emoji",               re.compile(
        r'[\U0001F300-\U0001F9FF'
        r'\U0001FA00-\U0001FAFF'
        r'\U00002600-\U000027BF'
        r'\U0001F000-\U0001F02F]'
    )),

    # Números isolados (linhas / tokens que são só dígitos)
    ("standalone_number",   re.compile(r'(?<!\w)\d{5,}(?!\w)')),

    # Marcadores de wiki / markdown  == Título == , [[ ]] , {{ }}
    ("wiki_markup",         re.compile(r'={2,}.*?={2,}||$|$.*?$|$||\{\{.*?\}\}')),

    # XML / comentários HTML  <!-- ... -->
    ("html_comment",        re.compile(r'<!--.*?-->', re.DOTALL)),

    # Sequências de caracteres não-ASCII repetidos (possível lixo de encoding)
    ("repeated_nonascii",   re.compile(r'[^\x00-\x7F]{6,}')),
]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def compute_utf8_mean(text: str) -> float:
    """Calcula a média dos codepoints / bytes UTF-8 do texto,
    excluindo CHARS_TO_REMOVE, usando a mesma lógica do pipeline."""
    cleaned = ''.join(c for c in text if c not in CHARS_TO_REMOVE)
    signal  = text_to_signal(cleaned)
    if len(signal) == 0:
        return 0.0
    return float(np.mean(signal))


def find_anomalies(text: str) -> list[dict]:
    """
    Retorna lista de dicts com informações sobre cada anomalia encontrada.
    Cada dict: {pattern_name, match_value, position}
    """
    findings = []
    for name, pattern in ANOMALY_PATTERNS:
        for m in pattern.finditer(text):
            findings.append({
                "pattern"  : name,
                "match"    : m.group(0)[:120],   # trunca para não explodir o CSV
                "position" : m.start(),
            })
    return findings


def load_texts_from_db(db_path: Path) -> list[dict]:
    """
    Lê todos os registros da tabela principal do banco.
    Retorna lista de dicts com: id, idioma, conteudo
    """
    conn   = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Descobre a primeira tabela disponível (mesma lógica do dataset_loader)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    if not tables:
        conn.close()
        raise RuntimeError("Nenhuma tabela encontrada no banco de dados.")

    table    = tables[0]
    col_text = "conteudo_uma_quebra" if USAR_CONTEUDO_TRATADO else "conteudo"
    col_lang = "idioma"

    print(f"Tabela usada  : {table}")
    print(f"Coluna de texto: {col_text}")

    # Tenta ler com coluna de id; se não existir, usa rowid
    try:
        cursor.execute(f"SELECT id, {col_lang}, {col_text} FROM {table}")
    except sqlite3.OperationalError:
        cursor.execute(f"SELECT rowid, {col_lang}, {col_text} FROM {table}")

    rows = cursor.fetchall()
    conn.close()

    records = []
    for row_id, idioma, conteudo in rows:
        if conteudo:
            records.append({
                "id"      : row_id,
                "idioma"  : idioma or "??",
                "conteudo": conteudo,
            })
    return records


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    start = datetime.now()

    print("=" * 60)
    print("INSPEÇÃO DE CORPUS — DETECÇÃO DE ANOMALIAS")
    print("=" * 60)
    print(f"Banco de dados : {DATABASE_PATH}")
    print(f"Saída CSV      : {OUTPUT_CSV}")
    print(f"Início         : {start.strftime('%d/%m/%Y %H:%M:%S')}")
    print()

    # 1. Carrega textos
    print("Carregando textos do banco...")
    records = load_texts_from_db(DATABASE_PATH)
    print(f"Total de registros carregados: {len(records)}")
    print()

    # 2. Analisa cada texto
    anomalies_found = []
    total_with_anomaly = 0

    print("Analisando textos...")
    for idx, rec in enumerate(records):
        if idx % 1000 == 0:
            print(f"  Processando registro {idx}/{len(records)}...", end="\r")

        findings = find_anomalies(rec["conteudo"])

        if not findings:
            continue

        total_with_anomaly += 1
        utf8_mean = compute_utf8_mean(rec["conteudo"])

        # Uma linha por padrão de anomalia encontrado
        patterns_seen = set()
        for f in findings:
            key = (rec["id"], f["pattern"])
            if key in patterns_seen:
                continue                    # evita duplicatas do mesmo padrão no mesmo texto
            patterns_seen.add(key)

            anomalies_found.append({
                "id"          : rec["id"],
                "idioma"      : rec["idioma"],
                "anomalia"    : f["pattern"],
                "exemplo"     : f["match"].replace("\n", "\\n").replace("\r", "\\r"),
                "posicao"     : f["position"],
                "tamanho"     : len(rec["conteudo"]),
                "media_utf8"  : round(utf8_mean, 4),
            })

    print(f"\nTextos com anomalias : {total_with_anomaly} / {len(records)}")
    print(f"Total de ocorrências : {len(anomalies_found)}")
    print()

    # 3. Grava CSV
    if anomalies_found:
        fieldnames = ["id", "idioma", "anomalia", "exemplo", "posicao", "tamanho", "media_utf8"]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(anomalies_found)
        print(f"CSV gerado com sucesso: {OUTPUT_CSV}")
    else:
        print("Nenhuma anomalia encontrada. CSV não gerado.")

    # 4. Resumo por tipo de anomalia
    if anomalies_found:
        print()
        print("--- Resumo por tipo de anomalia ---")
        from collections import Counter
        counter = Counter(row["anomalia"] for row in anomalies_found)
        for pattern_name, count in counter.most_common():
            print(f"  {pattern_name:<30} {count:>6} ocorrência(s)")

    # 5. Resumo por idioma
    if anomalies_found:
        print()
        print("--- Resumo por idioma ---")
        from collections import Counter
        lang_counter = Counter(row["idioma"] for row in anomalies_found)
        for lang, count in lang_counter.most_common():
            print(f"  {lang:<10} {count:>6} ocorrência(s)")

    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print()
    print(f"Tempo total: {elapsed:.1f} segundos")
    print(f"Término    : {end.strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()