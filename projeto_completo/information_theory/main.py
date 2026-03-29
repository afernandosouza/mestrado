"""
information_theory/main.py

Pipeline completo da 2ª etapa:
  1. Extração de características (WPT + Teoria da Informação)
  2. Visualização e análise de separabilidade

Executar a partir da raiz do projeto:
    python -m information_theory.main
"""

import sys
from pathlib import Path

# --------------------------------------------------------------------
# Ajuste do sys.path
# Sobe dois níveis a partir de baseline_reproduction/:
#   baseline_reproduction/ -> projeto_completo/
# Assim todos os módulos do projeto ficam acessíveis diretamente
# --------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]  # projeto_completo/
sys.path.insert(0, str(ROOT_DIR))

from information_theory.it_features import extract_information_theoretic_features
from information_theory.visualization import run_visualization


def main():
    extract_information_theoretic_features()
    run_visualization()


if __name__ == "__main__":
    main()