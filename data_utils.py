import torch
import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data

import config


# NumPy est une bibliothèque pour le calcul numérique en Python, elle offre des fonctions pour travailler avec des tableaux multidimensionnels.
# Pandas est une bibliothèque qui offre des structures de données et des outils d'analyse de données, facilitant la manipulation et l'analyse de datasets
# SciPy est une bibliothèque scientifique qui inclut des modules pour la manipulation de matrices
# torch.utils.data est Un module qui offre des utilitaires pour manipuler les datasets

def load_all(test_num=100):
	""" We load all the three file here to save time in each epoch. """
	train_data = pd.read_csv(config.train_rating, sep='\t', header=None, names=['user', 'item'], usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

	# fonction de panda permettant de lire un fichier csv, les valeurs sont séparées par des tabulations (espace), pas d'entête, nomme les colonnes 0 et 1 et spécifie les types"


	user_num = train_data['user'].max() + 1   #nbre d'users = dernier+1
	item_num = train_data['item'].max() + 1
	train_data = train_data.values.tolist()   #transforme en une liste
	

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)  # création d'une matrice de dimension (user_num, item_num)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0          # à gauche les users, à droite les items et 1.0 représente une intéraction		

	test_data = []
	with open(config.test_negative, 'r') as f:  # Ouvre le fichier en mode lecture  
		line = f.readline()                  
		while line != None and line != '':
			a = line.split('\t')
			test_data.append([eval(a[0])[0], eval(a[0])[1]])
			for i in a[1:]:
				test_data.append([u, int(i)])
			line = f.readline()
	return train_data, test_data, user_num, item_num, train_mat


#Cette fonction charge les données d'entraînement à partir d'un fichier CSV, les organise en une liste, puis crée une matrice creuse représentant les interactions user-item.
#Elle charge également des données de test à partir d'un fichier en ajoutant des exemples négatifs pour chaque utilisateur, et retourne toutes ces informations.






#Une feature représente une paire utilisateur-item





class NCFData(data.Dataset):     #hérite de torch.utils.data.Dataset
	def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
		super(NCFData, self).__init__()
		
		"""  the labels are only used when training, so we 
			add them in the ng_sample() function.
		"""
		
		self.features_ps = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng     #nbre d'échantillons négatifs
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]  #Initialise l'attribut labels avec une liste de zéros de  même longueur que les features





	def ng_sample(self):          #génère des échantillons négatifs lors de l'entraînement du modèle
                
		assert self.is_training 'no need to sampling sinon'
		self.features_ng = []  #liste vide pour stocker les caractéristiques des échantillons négatifs.
		for x in self.features_ps:      #La boucle parcourt chaque paire user-item positive dans les caractéristiques d'entraînement.
			u = x[0] #u = x[0]: Extrait l'ID du user de la paire positive.

			for t in range(self.num_ng): #génère self.num_ng échantillons négatifs pour chaque paire positive.
				j = np.random.randint(self.num_item)   #Génère un id d'item négatif de manière aléatoire.
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)   #Si la paire est déjà dans la matrice d'entraînement, génère un nouvel identifiant d'item négatif
				self.features_ng.append([u,j])

		labels_ps=[1 for _ in range(len(self.features_ps))] #Crée une liste de 1 de la même longueur que les caractéristiques positives
		labels_ng=[0 for _ in range(len(self.features_ng))] #Crée une liste de 0 de la même longueur que les caractéristiques négatives

		self.features_fill = self.features_ps + self.features_ng #Crée une liste combinant les deux
		self.labels_fill = labels_ps + labels_ng #de même pour les labels





		

	def __len__(self):
		return (self.num_ng + 1) * len(self.labels) #taille de la dataset



	

	def __getitem__(self, i):   #i : indice de l'échantillon
		features = self.features_fill if self.is_training
					else self.features_ps
		labels = self.labels_fill if self.is_training
					else self.labels

		user = features[i][0]
		item = features[i][1]
		label = labels[i]
		return user, item ,label

#cette fonction retourne un triplet constitué d'un user, d'un item et d'une étiquette, en fonction de l'indice fourni.
#Elle prend en compte le mode d'entraînement pour sélectionner les features et labels appropriés.		
