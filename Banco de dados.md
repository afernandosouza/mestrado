# Database Structure

This project uses a SQLite database named `banco_texto.db`, which contains the texts and metadata required for the experiments in:

- **Stage 1** – Baseline reproduction of Hassanpour et al. (2021)  
- **Stage 2** – Information-theoretic analysis and visualization

The database can be downloaded from the following link:

- Download: https://drive.google.com/file/d/1wcJy5F610dQDNhMSStlgFV2xuGPpjH5e/view?usp=sharing

## Main Table

The database has a single main table (one logical table), with the following structure:

- `rowid` (INTEGER, implicit SQLite key)  
  Internal numeric identifier of each record. Used as an index.

- `idioma` (TEXT, NOT NULL)  
  Language code of the text (e.g., `en`, `pt`, `ar`, `fa`, etc.).

- `conteudo` (TEXT, NOT NULL)  
  Original full text, as collected, including multiple consecutive line breaks and all characters.

- `conteudo_uma_quebra` (TEXT, NULLABLE)  
  Normalized version of the text where sequences of two or more consecutive line breaks are reduced to a single line break (`\n`).  
  This field is used when `USAR_CONTEUDO_TRATADO = True` in `config.py`.

- `media_utf8` (REAL, NULLABLE)  
  Mean of the UTF-8 codes of the original text (`conteudo`), computed after removing the characters specified in `CHARS_TO_REMOVE` (by default: `@`, `-`, `+`, `#`).  
  This is the feature used in the K-means clustering step of the baseline method (Hassanpour reproduction).

- `media_utf8_uma_quebra` (REAL, NULLABLE)  
  Mean of the UTF-8 codes of the normalized text (`conteudo_uma_quebra`), computed with the same removal rule as `media_utf8`.  
  This field allows comparison of the impact of line-break normalization on the clustering and subsequent analysis.

## Usage in the Experiments

- The function `load_dataset_sqlite` (module `data.dataset_loader`) reads:
  - the text field (`conteudo` or `conteudo_uma_quebra`), according to `USAR_CONTEUDO_TRATADO`;  
  - the corresponding UTF-8 mean field (`media_utf8` or `media_utf8_uma_quebra`).

- In **Stage 1 (baseline reproduction)**:
  - `media_utf8` (or `media_utf8_uma_quebra`) is used as the input feature for K-means clustering;  
  - the selected text field is converted into a time series for extraction of the 32 WPT energy features.

- In **Stage 2 (information-theoretic analysis)**:
  - the same text field is used to compute:
    - Shannon entropy (H),
    - normalized entropy (H_norm),
    - sub-band energy entropy (H_sub),
  producing a combined feature vector (WPT + information-theoretic metrics) per text, which is then used for visualization and cluster separability analysis, without training any classifier.