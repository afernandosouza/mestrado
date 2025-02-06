#encoding: utf-8
import codecs
from slugify import slugify

if __name__ == '__main__':
	idioma_atual = ''
	try:
		#Navega pelos arquivo de textos removendo os espaços em branco e quebras de linha 
		
		with open('IDIOMAS.txt', 'r', encoding='UTF-8') as idiomas:
			for idioma in idiomas.readlines():
				idioma_atual = idioma
				print('limpando arquivo %s...' % idioma)
				dados = idioma.split(' - ')
				arq = open('%s_%s.txt' % (slugify(dados[0].strip()), dados[1].strip().replace('\n','')), 'r', encoding='UTF-8')
				
				new_text = '';
				for linha in arq.readlines():
					if linha not in new_text:
						new_text += '%s\n' % linha

				clean_arq = codecs.open('%s_%s_clean.txt' % (slugify(dados[0].strip()), dados[1].strip().replace('\n','')), 'w', 'UTF-8')
				#new_text = arq.read()
				clean_arq.write(new_text.lower()
					#.replace('\n','')
					#.replace(';','')
					.replace('wikipédia','')
					.replace('mover','')
					.replace('barra','')
					.replace('lateral','')
					.replace('páginas','')
					.replace('-','')
					.replace('_','')
					.replace('•','')
					.replace('imprimirexportar','')
					.replace('especiaishiperligação','')
					.replace('permanenteinformações','')
					)
				clean_arq.close()
				arq.close()
				print('Arquido %s limpo' % dados[0])

			print('Arquivos limpos\n')
						
	except Exception as e:
		print('Erro ao processar o idioma %s' % idioma_atual)
		print(e)
		raise