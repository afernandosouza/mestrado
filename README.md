# Repositório do Mestrado – Metodologias

Este repositório contém dois componentes principais relacionados à identificação de idiomas em texto:

1. **Reprodução da baseline de Hassanpour et al. (2021)** – `main.py`  
2. **Interface gráfica para experimentos em planos de Teoria da Informação** – `gui_experiment.py`

A seguir são descritas, de forma resumida e objetiva, as metodologias implementadas em cada componente.

---

## 1. Reprodução da Baseline (main.py)

### 1.1. Objetivo

Reproduzir e estender o método de *Language Identification* baseado em processamento de sinais descrito em:

> Hassanpour, H.; AlyanNezhadi, M. M.; Mohammadi, M.  
> *A Signal Processing Method for Text Language Identification.*  
> International Journal of Engineering, 34(6), 1413–1418, 2021.

A pipeline é aplicada a um corpus multilíngue armazenado em SQLite e avalia o impacto de espaçamentos artificiais entre palavras, bem como o uso opcional de *features* de Teoria da Informação.

### 1.2. Dados

- Os textos são carregados a partir de um banco SQLite via:
  - `data.dataset_loader.load_dataset_sqlite(DATABASE_REF)`
- Retornos principais:
  - `texts`: lista de textos;
  - `labels`: rótulos inteiros (uma classe por idioma);
  - `unique_langs`: lista de idiomas únicos;
  - `raw_labels`: rótulos textuais originais (códigos de idioma).

### 1.3. Espaçamento Artificial

Para seguir a proposta de Hassanpour et al., o código aplica espaçamentos artificiais entre as palavras:

- Função utilizada: `spacing_experiment.apply_spacing`
- Níveis de espaçamento controlados por `SPACING_LEVELS` (em `config.py`).
- Utiliza os espaçamentos 1, 5, 7 e 12, é gerado um novo conjunto de textos `spaced_texts`, que é então utilizado na etapa de treinamento/teste.

### 1.4. Divisão Treino/Teste

É utilizada uma divisão estratificada em treino e teste para **cada nível de espaçamento**, preservando a distribuição de idiomas:

- Função auxiliar: `custom_train_test_split`
  - Gera índices de treino/teste com `train_test_split` (estratificado em `labels`);
  - Aplica-os a:
    - textos com espaçamento (`X_train_spaced`, `X_test_spaced`);
    - rótulos numéricos (`y_train`, `y_test`);
    - rótulos brutos por idioma (`raw_train_labels`, `raw_test_labels`).

A proporção treino/teste é definida em `TEST_SPLIT`, configurada em `config.py`.

### 1.5. Pipeline de LID

O núcleo da metodologia está em `LIDPipeline`, que combina:

- Um modelo de **clusterização** de textos (`ClusterModel`, ver abaixo);
- Um classificador MLP (implementado dentro da pipeline);
- Opcionalmente, *features* de Teoria da Informação pré-computadas.

#### 1.5.1. Clusterização baseada em média UTF-8

A classe `ClusterModel` (arquivo `cluster_model.py`) implementa a estratégia de Hassanpour et al.:

1. Converte cada texto em uma série de inteiros via `text_to_signal` (codepoints Unicode).
2. Remove caracteres específicos definidos em `CHARS_TO_REMOVE`.
3. Calcula a média dos códigos (`mean UTF-8`) para cada texto.
4. Normaliza a *feature* escalar com `StandardScaler`.
5. Aplica `KMeans` com:
   - `n_clusters = N_CLUSTERS`;
   - `random_state = RANDOM_STATE`;
   - `n_init = N_INIT_KMEANS`.

Assim, cada texto é mapeado a um único valor (média), que é clusterizado. Esse rótulo de *cluster* pode ser usado como *feature* de entrada para a MLP da pipeline.

#### 1.5.2. Features de Teoria da Informação (opcionais)

- `USE_TI_FEATURES` controla se *features* de TI serão utilizadas.
- São carregadas uma única vez via:
  - `data.ti_features_loader.load_ti_features_from_db(DATABASE_TI_REF)`
- Essas *features* (por exemplo, coordenadas em planos Complexidade–Entropia ou Fisher–Shannon) são associadas a cada idioma (`raw_labels`) e injetadas na `LIDPipeline`:
  - No `fit`, são passadas junto com os textos de treino;
  - No `predict`, o pipeline recebe o `raw_label` (idioma esperado do texto) para buscar as *features* correspondentes.

As *features* de TI funcionam como atributos adicionais de alto nível, agregados por idioma.

### 1.6. Treinamento e Predição

Para cada nível de espaçamento:

1. Cria-se uma instância de `LIDPipeline`:
   ```python
   pipeline = LIDPipeline(N_CLUSTERS, ti_features_data if use_ti_features_current_run else None)
   ```
2. Treina-se o modelo:
   ```python
   pipeline.fit(X_train_spaced, y_train, raw_train_labels if use_ti_features_current_run else None)
   ```
3. Prediz-se o idioma de cada texto de teste:
   ```python
   preds.append(pipeline.predict(text_spaced, raw_test_labels[i] if use_ti_features_current_run else None))
   ```
4. Calcula-se a acurácia por execução:
   ```python
   acc = np.mean(np.array(preds) == np.array(y_test))
   ```

Cada experimento é repetido `N_RUNS` vezes, acumulando as acurácias em `acc_runs`.

### 1.7. Métricas, Estatísticas e Saída

- Para cada nível de espaçamento, são calculadas estatísticas agregadas:
  - `evaluation.statistics.compute_statistics(acc_runs)`  
    (média, desvio padrão, intervalo de confiança, etc.)
- Resultados são:
  - Logados via `utils.logger`;
  - Salvos em disco via `evaluation.save_results.save_results`;
  - Plotados em gráficos (e.g., acurácia vs. espaçamento) via `evaluation.plot_results.plot_results`;
  - Matriz de confusão final salva com `evaluation.confusion.save_confusion`.

Além disso, o script monitora uso de CPU e memória durante as execuções com:

- `utils.system_info.SystemMonitor`,
- Relatando ao final com `print_and_log_monitor_results`.

---

## 2. Interface Gráfica para Experimentos de Teoria da Informação (gui_experiment.py)

### 2.1. Objetivo

Fornecer uma interface interativa para:

- Visualizar idiomas em:
  - Plano Complexidade–Entropia (Bandt–Pompe);
  - Plano Fisher–Shannon;
- Experimentar com:
  - Dimensão de imersão `m` e atraso `τ`;
  - Uso de sinal original de codepoints ou sinal transformado por *wavelet*;
- Avaliar separabilidade entre idiomas com métricas baseadas apenas em TI;
- Comparar textos novos com clusters de idioma (detecção aproximada de idioma no plano de TI).

### 2.2. Dados e Seleção de Idiomas

- Carregamento via `data.dataset_it.load_dataset_it(DATABASE_REF)`, retornando:
  - textos, rótulos numéricos, códigos de idioma (`lang_codes`), rótulos brutos, nomes completos;
- A GUI cria *checkboxes* para cada `lang_code`, permitindo selecionar um ou mais idiomas para análise conjunta.

### 2.3. Sinal: Original vs Wavelet

A série temporal base é o sinal de codepoints Unicode do texto:

- `signal_processing.text_signal.text_to_signal(text)`

A GUI permite duas opções de sinal:

1. **Original (codepoints)**  
   - Usa diretamente a sequência de inteiros gerada pelos codepoints.

2. **Wavelet (db4, 5 níveis)**  
   - Aplica transformada wavelet discreta (DWT) com família Daubechies 4 (`db4`) usando `pywt`:
     - Decomposição até `level = 5` (ajustado se o sinal for curto);
     - Seleciona coeficientes de detalhe `D_k` de um nível especificado (`wavelet_level_var`);
     - Normaliza coeficientes (média 0, desvio padrão 1);
   - A função `_get_wavelet_signal` encapsula essa lógica.

O tipo de sinal é controlado por *radiobuttons* na GUI.

### 2.4. Plano Complexidade–Entropia (Bandt–Pompe)

Para cada texto e idioma selecionado:

1. Gera-se o sinal (original ou wavelet).
2. Se o tamanho do sinal satisfaz `len(signal) >= m * τ`:
   - Calcula-se:
     - Entropia de permutação normalizada |$H_s$|;
     - Complexidade estatística |$C$|;
   - via `information_theory.bandt_pompe_complexity`.

Essa função:

- Constrói a distribuição de padrões ordinais (permutations) com dimensão `m` e atraso `τ`;
- Calcula:
  - |$ H_s = -\frac{1}{\log(m!)} \sum p_\pi \log p_\pi $|;
  - Complexidade estatística |$ C = Q_J \cdot H_s $|, onde |$Q_J$| é uma divergência de Jensen normalizada em relação à distribuição uniforme.

A GUI:

- Plota cada idioma em um diagrama |$ H_s \times C $|:
  - Pontos individuais (um por texto);
  - Centroide (média de |$H_s, C$|);
  - Elipse de pertencimento (e.g. ±1σ);
- Opcionalmente, plota curvas teóricas de referência (curva de complexidade máxima e curva de complexidade mínima) quando a normalização está ativa.

### 2.5. Plano Fisher–Shannon

Analogamente, para cada texto:

1. Gera-se o sinal (original ou wavelet).
2. Calcula-se:
   - Entropia de permutação normalizada |$H_s$|;
   - Informação de Fisher normalizada |$F$|;
   - via `information_theory.fisher_shannon_experiment.compute_hs_f`.

A GUI plota |$ H_s \times F $| com:

- Pontos por texto,
- Centróides por idioma,
- Elipses de pertencimento (e.g. ±2σ).

### 2.6. Cache e Reprodutibilidade dos Experimentos

Para evitar recomputação, os resultados por idioma/plano/parâmetros são cacheados:

- Uso de `information_theory.experiment_cache.save_experiment` e `load_experiment`.
- A chave de cache inclui:
  - idioma (`lang_code`),
  - espaço (`"bp"` ou `"fs"`),
  - `m`, `τ`,
  - tipo de sinal (`original` ou `wavelet`) e nível de detalhe,
  - indicador de normalização.

Isso permite reproduzir rapidamente visualizações com parâmetros já usados.

### 2.7. Avaliação da Separabilidade entre Idiomas

A GUI implementa uma avaliação puramente baseada em TI, sem modelos supervisionados:

1. Para o plano atual (Bandt–Pompe ou Fisher–Shannon), compila todos os pontos:
   - |$ X = [(H_s, C/F)] $|;
   - rótulos numéricos por idioma.
2. Calcula:
   - **Índice de Silhueta** (via `sklearn.metrics.silhouette_score`):
     - Mede quão bem os clusters de idiomas estão separados (0–1).
   - **Razão intra/inter-distância (R)**:
     - Distância média entre pontos do mesmo idioma (intra);
     - Distância média entre centróides de idiomas distintos (inter);
     - |$ R = \frac{\text{distância intra média}}{\text{distância inter média}} $| (quanto menor, melhor).
   - **Acurácia por centróide mais próximo**:
     - Classifica cada ponto pelo centróide de idioma mais próximo (distância Euclidiana);
     - Calcula a proporção de acertos.

Essas métricas são apresentadas em uma janela informativa (“Avaliar separabilidade (TI)”).

### 2.8. Comparação de Texto Novo vs Idioma de Referência

A GUI permite colar um texto novo e verificar se ele é compatível com um idioma de referência, diretamente nos planos de TI:

1. O usuário seleciona ao menos um idioma e gera o plano (BP ou FS).
2. O primeiro idioma selecionado é tomado como referência.
3. A partir das estatísticas do idioma de referência:
   - |$\mu_{Hs}, \mu_{C/F}$|, |$\sigma_{Hs}, \sigma_{C/F}$|, tipo de sinal e parâmetros.
4. Para o texto novo:
   - Converte para sinal (original ou wavelet, igual ao idioma de referência).
   - Adapta `m` e `τ` com base no comprimento do sinal (parâmetros adaptativos para textos curtos).
   - Calcula |$H_s$| e |$C$| ou |$F$|.
   - Calcula distância normalizada ao centróide:
     - |$ d = \sqrt{\left(\frac{H_s - \mu_{Hs}}{\sigma_{Hs}}\right)^2 + \left(\frac{Y - \mu_Y}{\sigma_Y}\right)^2} $|.
   - Compara com um limiar adaptativo (menor para textos longos, maior para textos curtos):
     - Se |$ d \leq \text{threshold} $|: o texto é considerado **pertencente** ao idioma;
     - Caso contrário, é marcado como fora da região esperada.

O ponto do texto novo é plotado como uma estrela (*), em verde (pertence) ou vermelho (não pertence).

### 2.9. Listagem de Outliers

A GUI implementa uma ferramenta para inspecionar textos que se comportam como outliers no plano atual:

1. Para cada idioma selecionado:
   - Calcula a distância de cada ponto ao centróide em unidades de σ (como na comparação acima).
2. Marca como outlier textos cuja distância excede um limiar fixo (por padrão, 5σ).
3. Abre uma janela com:
   - Lista de idiomas;
   - Para cada idioma, textos outliers (trechos) e sua distância em σ.

Isso permite identificar textos atípicos (ruído, metadados, textos curtos demais, etc.) que afetam a forma do cluster.

### 2.10. Exportação de Gráficos e Dados

A GUI permite:

- **Exportar gráfico (PNG)**:
  - Nome inclui:
    - tipo de plano (`bp` ou `fs`),
    - idiomas selecionados,
    - `m`, `τ`,
    - indicação de normalização,
    - tipo de sinal e nível wavelet (se aplicável).
- **Exportar dados (CSV)**:
  - Para cada idioma, salva colunas `Hs_idioma`, `C_idioma` ou `F_idioma`;
  - Antes do cabeçalho do CSV, escreve linhas de comentários com:
    - centróides e desvios padrão em |$H_s$| e C/F,
    - indicação se os dados estão normalizados.

### 2.11. Documentação embutida (Metodologia e Cálculos)

O botão “Ver cálculos” abre uma janela explicativa com:

- Descrição da construção da série de codepoints;
- Descrição do uso de Wavelet (db4, 5 níveis);
- Fórmulas de:
  - Entropia de permutação;
  - Complexidade estatística;
  - Informação de Fisher;
- Interpretação dos planos e da elipse de pertencimento.

---

## 3. Requisitos Principais

- Python 3.x
- Bibliotecas:
  - `numpy`, `pandas`, `scikit-learn`, `scipy`
  - `matplotlib`, `tkinter`
  - `pywt` (PyWavelets)
  - Outras dependências internas do projeto (`config`, `data`, `information_theory`, `signal_processing`, `evaluation`, `utils`).

---

## 4. Execução

### 4.1. Executar a reprodução da baseline

```bash
python main.py
```

- Roda o experimento completo para todos os níveis de espaçamento definidos em `config.py`.
- Salva logs, métricas, gráficos e matrizes de confusão.

### 4.2. Executar a GUI de experimentos de TI

```bash
python gui_experiment.py
```

- Abre a interface gráfica;
- Permite selecionar idiomas, ajustar parâmetros e visualizar os planos Complexidade–Entropia e Fisher–Shannon, além de comparar textos novos.

---

# Estrutura do Banco de Dados

Este projeto utiliza um banco SQLite chamado `banco_texto.db`, que contém os textos e metadados necessários para os experimentos em:

- **Etapa 1** – Reprodução da baseline de Hassanpour et al. (2021)  
- **Etapa 2** – Análise e visualização baseadas em Teoria da Informação

O banco pode ser baixado pelo seguinte link:

- Download: https://drive.google.com/file/d/1wcJy5F610dQDNhMSStlgFV2xuGPpjH5e/view?usp=sharing

## Tabela Principal

O banco possui uma única tabela principal (uma tabela lógica), com a seguinte estrutura:

- `rowid` (INTEGER, chave implícita do SQLite)  
  Identificador numérico interno de cada registro. Usado como índice.

- `idioma` (TEXT, NOT NULL)  
  Código de idioma do texto (por exemplo, `en`, `pt`, `ar`, `fa`, etc.).

- `conteudo` (TEXT, NOT NULL)  
  Texto original completo, conforme coletado, incluindo quebras de linha consecutivas e todos os caracteres.

- `conteudo_uma_quebra` (TEXT, NULLABLE)  
  Versão normalizada do texto em que sequências de duas ou mais quebras de linha consecutivas são reduzidas a uma única quebra de linha (`\n`).  
  Este campo é utilizado quando `USAR_CONTEUDO_TRATADO = True` em `config.py`.

- `media_utf8` (REAL, NULLABLE)  
  Média dos códigos UTF-8 do texto original (`conteudo`), calculada após remover os caracteres especificados em `CHARS_TO_REMOVE` (por padrão: `@`, `-`, `+`, `#`).  
  Esta é a feature usada na etapa de clusterização K-means do método baseline (reprodução de Hassanpour).

- `media_utf8_uma_quebra` (REAL, NULLABLE)  
  Média dos códigos UTF-8 do texto normalizado (`conteudo_uma_quebra`), calculada com a mesma regra de remoção de `media_utf8`.  
  Este campo permite comparar o impacto da normalização de quebras de linha na clusterização e na análise subsequente.

## Uso nos Experimentos

- A função `load_dataset_sqlite` (módulo `data.dataset_loader`) lê:
  - o campo de texto (`conteudo` ou `conteudo_uma_quebra`), conforme `USAR_CONTEUDO_TRATADO`;  
  - o campo de média UTF-8 correspondente (`media_utf8` ou `media_utf8_uma_quebra`).

- Na **Etapa 1 (reprodução da baseline)**:
  - `media_utf8` (ou `media_utf8_uma_quebra`) é usada como feature de entrada para a clusterização K-means;  
  - o campo de texto selecionado é convertido em série temporal para extração das 32 features de energia WPT.

- Na **Etapa 2 (análise via Teoria da Informação)**:
  - o mesmo campo de texto é usado para calcular:
    - entropia de Shannon (H),
    - entropia normalizada (H_norm),
    - entropia de energia em sub-bandas (H_sub),
  produzindo um vetor de features combinado (WPT + métricas de Teoria da Informação) por texto, que é então usado para visualização e análise de separabilidade de clusters, sem treinar nenhum classificador.