from datetime import datetime
import requests
import time
import json
import os

# Configuração do User-Agent para cumprir a política da Wikipédia
USER_AGENT = "MeuScriptMestradoUfalPPGI/1.0 (afernandosouza@gmail.com)"
NUMERO_TEXTOS = 1
NUMERO_TENTATIVAS = 3000

def obter_textos_aleatorios(limite, idioma):
    try:
        """
        Obtém textos aleatórios da Wikipédia no idioma especificado.
        
        Parâmetros:
            limite (int): Número de textos a serem obtidos.
            idioma (str): Código do idioma da Wikipédia (ex.: 'pt', 'en', 'es').
            
        Retorna:
            list: Lista de dicionários contendo título e conteúdo dos textos.
        """
        print(f'Obtendo textos da wikipedia para o idioma {idioma}...')
        textos = []
        base_url = f"https://{idioma}.wikipedia.org/w/api.php"  # URL da API para o idioma escolhido
        limite_atingido = False
        contador = 0
        conteudo5000 = ''

        while not limite_atingido:
            # Faz a solicitação à API para obter uma página aleatória
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnlimit": 1,  # Um artigo por vez
                "rnnamespace": 0,  # Somente artigos (remover para incluir categorias, etc.)
            }
            headers = {"User-Agent": USER_AGENT}

            response = requests.get(base_url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                for page in data["query"]["random"]:
                    page_id = page["id"]
                    title = page["title"]

                    # Obtém o conteúdo do artigo
                    print(f'Obtendo conteúdo {contador + 1} do idioma {idioma}')
                    conteudo = obter_conteudo_pagina(page_id, base_url)
                    contador += 1
                    
                    if conteudo:
                        if len(conteudo) < 5000:
                            conteudo5000 += f'\n\n{conteudo}'
                        else:
                            conteudo5000 = conteudo

                    if len(conteudo5000) >= 5000:
                        textos.append({"title": title, "content": conteudo5000, "tamanho": len(conteudo5000)})
                        print('texto adicionado', len(conteudo5000))
                        conteudo5000 = ''
                        
            # Adiciona um pequeno atraso para evitar sobrecarregar a API
            time.sleep(0.5)

            if len(textos) >= limite or contador == NUMERO_TENTATIVAS:
                limite_atingido = True
                conteudo5000 = ''
    except Exception as e:
        print(e)
        raise

    return textos

def obter_conteudo_pagina(page_id, base_url):
    """
    Obtém o conteúdo de uma página da Wikipédia pelo ID.
    
    Parâmetros:
        page_id (int): ID da página.
        base_url (str): URL base da API da Wikipédia.
        
    Retorna:
        str: Conteúdo em texto simples da página.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "pageids": page_id,
    }
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        page = data["query"]["pages"].get(str(page_id), {})
        return page.get("extract")
    return None

def carrega_idiomas():
    try:
        with open('IDIOMA.txt', 'r', encoding="utf-8") as idiomas:
            return [i.split(' - ')[1].strip() for i in idiomas.readlines()]
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    try: 
        # Executa o código para obter 1000 textos aleatórios em um idioma especificado
        #idioma = "en"  # Substitua pelo idioma desejado, ex.: 'en', 'es', 'fr'
        limite = NUMERO_TEXTOS  # Quantidade de textos a serem obtidos
        hoje = datetime.now()

        for idioma in carrega_idiomas():
            textos_aleatorios = obter_textos_aleatorios(limite, idioma)

            # Salva os textos em um arquivo JSON
            arquivo_saida = os.path.join(os.getcwd(),'TEXTOS',f"textos_{idioma}_{hoje.strftime('%d%m%Y%H%M%S')}.json")
            with open(arquivo_saida, "w", encoding="utf-8") as f:
                json.dump(textos_aleatorios, f, ensure_ascii=False, indent=4)

            print(f"Total de textos obtidos para o idioma {idioma}: {len(textos_aleatorios)}")
            print(f"Textos salvos em: {arquivo_saida}")
    except Exception as e:
        print(e)
        raise
