"""
migrar_coluna_quebra.py

Adiciona e popula as colunas:
  - conteudo_uma_quebra   : conteudo com múltiplas quebras reduzidas a 1
  - media_utf8_uma_quebra : média UTF-8 do conteudo_uma_quebra
    (mesma lógica do artigo: remove @, -, +, # antes de calcular)

Executar uma única vez:
    python migrar_coluna_quebra.py
"""

import warnings
warnings.filterwarnings('ignore')

import re
import sqlite3
from pathlib import Path

from config import DATABASE, CHARS_TO_REMOVE


# --------------------------------------------------------------------
# Funções auxiliares
# --------------------------------------------------------------------
def reduzir_quebras(conteudo: str) -> str:
    """Reduz 2+ quebras de linha consecutivas para exatamente 1."""
    return re.sub(r'\n{2,}', '\n', conteudo)


def calcular_media_utf8(conteudo: str) -> float:
    """
    Calcula a média dos codepoints UTF-8 do conteudo,
    removendo os caracteres em CHARS_TO_REMOVE.
    Mesma lógica usada para gerar o campo media_utf8 original.
    """
    vals = [ord(c) for c in conteudo if c not in CHARS_TO_REMOVE]
    return float(sum(vals) / len(vals)) if vals else 0.0


# --------------------------------------------------------------------
# Migração principal
# --------------------------------------------------------------------
def migrar_colunas(db_path: Path):

    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    # Descobre nome da tabela
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    TABLE = cur.fetchall()[0][0]
    print(f"Tabela: {TABLE}")

    # Verifica colunas existentes
    cur.execute(f"PRAGMA table_info({TABLE});")
    colunas_existentes = [row[1] for row in cur.fetchall()]
    print(f"Colunas existentes: {colunas_existentes}")

    # ------------------------------------------------------------
    # Cria colunas se ainda não existirem
    # ------------------------------------------------------------
    colunas_novas = {
        "conteudo_uma_quebra"   : "TEXT",
        "media_utf8_uma_quebra" : "REAL",
    }

    for coluna, tipo in colunas_novas.items():
        if coluna not in colunas_existentes:
            print(f"Criando coluna '{coluna}'...")
            cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN {coluna} {tipo}")
            conn.commit()
            print(f"  Coluna '{coluna}' criada.")
        else:
            print(f"  Coluna '{coluna}' já existe — será atualizada.")

    # ------------------------------------------------------------
    # Lê todos os registros
    # ------------------------------------------------------------
    cur.execute(f"SELECT rowid, conteudo FROM {TABLE};")
    rows = cur.fetchall()
    print(f"\nTotal de registros a processar: {len(rows)}")

    # ------------------------------------------------------------
    # Processa em lotes
    # ------------------------------------------------------------
    LOTE = 1000
    total_atualizado   = 0
    total_com_diferenca = 0

    for start in range(0, len(rows), LOTE):
        lote  = rows[start:start + LOTE]
        dados = []

        for rowid, conteudo in lote:
            if conteudo:
                conteudo_tratado = reduzir_quebras(conteudo)
                media_tratado    = calcular_media_utf8(conteudo_tratado)

                # Conta quantos registros tiveram alteração de fato
                if conteudo_tratado != conteudo:
                    total_com_diferenca += 1
            else:
                conteudo_tratado = conteudo
                media_tratado    = 0.0

            dados.append((conteudo_tratado, media_tratado, rowid))

        cur.executemany(f"""
            UPDATE {TABLE}
            SET conteudo_uma_quebra   = ?,
                media_utf8_uma_quebra = ?
            WHERE rowid = ?
        """, dados)
        conn.commit()

        total_atualizado += len(lote)
        if (start // LOTE) % 5 == 0:
            print(f"  Processados: {total_atualizado}/{len(rows)}")

        del lote, dados

    # ------------------------------------------------------------
    # Verificação final
    # ------------------------------------------------------------
    print(f"\nMigração concluída.")
    print(f"  Total processado          : {total_atualizado}")
    print(f"  Registros com diferença   : {total_com_diferenca} "
          f"({total_com_diferenca / total_atualizado * 100:.2f}%)")

    cur.execute(f"""
        SELECT COUNT(*) FROM {TABLE}
        WHERE conteudo_uma_quebra   IS NOT NULL
          AND media_utf8_uma_quebra IS NOT NULL
    """)
    preenchidos = cur.fetchone()[0]
    print(f"  Registros preenchidos     : {preenchidos}")

    # Comparação entre médias originais e tratadas
    cur.execute(f"""
        SELECT
            AVG(media_utf8)            AS media_original,
            AVG(media_utf8_uma_quebra) AS media_tratada,
            AVG(ABS(media_utf8 - media_utf8_uma_quebra)) AS diferenca_media
        FROM {TABLE}
        WHERE media_utf8 IS NOT NULL
          AND media_utf8_uma_quebra IS NOT NULL
    """)
    row = cur.fetchone()
    print(f"\nComparação de médias UTF-8:")
    print(f"  Média original  (media_utf8)           : {row[0]:.4f}")
    print(f"  Média tratada   (media_utf8_uma_quebra): {row[1]:.4f}")
    print(f"  Diferença média absoluta               : {row[2]:.6f}")

    conn.close()


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
if __name__ == "__main__":
    migrar_colunas(DATABASE)