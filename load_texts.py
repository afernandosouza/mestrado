import requests
from bs4 import BeautifulSoup as bs

if __name__ == '__main__':
	idioma_atual = ''
	try:
		#Abre o arqiovo IDIOMAS.txt e navega pelas páginas de todos os idiomas na wikipedia 
		
		with open('IDIOMAS.txt', 'r', encoding='UTF-8') as idiomas:
			for idioma in idiomas.readlines():
				idioma_atual = idioma
				print('gerando arquivo para o idioma %s...' % idioma)
				dados = idioma.split(' - ')
				DOMINIO = 'https://%s.wikipedia.org/' % dados[1].replace('\n','')
				conteudo = requests.get(DOMINIO, timeout=5, verify=False)
				if conteudo.status_code == 200:
					#Gera um arquivo para o idioma atual com o texto obtido
					arq = open('%s_%s.txt' % (dados[0], dados[1].replace('\n','')), 'w', encoding='UTF-8')
					site = bs(conteudo.text, "html.parser")
					arq.write(' '.join(''.join([div.text for div in site.find_all('div')]).replace('\n', '').split(" ")), encoding='UTF-8')
					arq.close()
				print('arquivo gerado\n')
						
	except Exception as e:
		print('Erro ao processar o idioma %s' % idioma_atual)
		raise