
if __name__ == '__main__':
	try:
		with open("load_texts.py") as file:
			exec(file.read())
		with open("clean_texts.py") as file:
			exec(file.read())
		with open("create_block.py") as file:
			exec(file.read())	
	except Exception as e:
		print(e)
		raise