import sys, os, re, platform
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import checkFasta

pPath = re.sub(r'codes$', '', os.path.split(os.path.realpath(__file__))[0])
sys.path.append(pPath)
from codes import readFasta
import numpy as np
import pandas as pd

def AAINDEX(fastas, **kw):
	if checkFasta.checkFasta(fastas) == False:
		print('Error: for "AAINDEX" encoding, the input fasta sequences should be with equal length. \n\n')
		return 0

	AA = 'ARNDCQEGHILKMFPSTWYV'

	fileAAindex = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\AAindex.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/AAindex.txt'
	with open(fileAAindex) as f:
		records = f.readlines()[1:]

	AAindex = []
	AAindexName = []
	for i in records:
		AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
		AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

	index = {}
	for i in range(len(AA)):
		index[AA[i]] = i

	encodings = []
	header = ['#']
	for pos in range(1, len(fastas[0][1]) + 1):
		for idName in AAindexName:
			header.append('SeqPos.' + str(pos) + '.' + idName)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		for aa in sequence:
			if aa == '-':
				for j in AAindex:
					code.append(0)
				continue
			for j in AAindex:
				code.append(j[index[aa]])
		encodings.append(code)

	return encodings


fastas = readFasta.readFasta(r"F:\python\KNN\test_A.txt")
kw=  {'path': r"F:\python\KNN",'train':r"F:\python\KNN\test_A.txt",'label':r"F:\python\KNN\label_A.txt",'order':'ACDEFGHIKLMNPQRSTVWY'}
result=AAINDEX(fastas, **kw)
data=result[1:]
data_new=np.matrix(data)
data_AAindex=data_new[:,1:]
data_AAindex_end=pd.DataFrame(data=data_AAindex)
data_AAindex_end.to_csv('AAindex.csv')
