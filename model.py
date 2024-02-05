import torch
import torch.nn as nn
import torch.nn.functional as F 


class NCF(nn.Module):  ##NCF hérite de nn.Module, une classe de base pour les modules en PyTorch
	def __init__(self, user_num, item_num, factor_num, num_layers, dropout, model, GMF_model=None, MLP_model=None): #constructeur
		super(NCF, self).__init__()
		"""
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		GMF_model: pre-trained GMF weights;  MLP_model: pre-trained MLP weights.
		
		# Les  "poids pré-entraînés" sont les paramètres ajustables appris sur des données antérieures, pouvant être utilisés pour initialiser le modèle NCF.
		# chaque utilisateur/article est représenté par un vecteur d'embedding de dimension factor_num
		# le modèle MLP contient num_layers
		# dropout: Chaque neurone a (dropout)% de chance d'être désactivé pendant chaque itération de l'entraînement. (appliqué entre les couches entièrement connectés)
		
                """
		
		self.dropout = dropout
		self.model = model
		self.GMF_model = GMF_model
		self.MLP_model = MLP_model

		self.embed_user_GMF = nn.Embedding(user_num, factor_num) #Crée une couche d'embedding pour les utilisateurs dans la partie GMF du modèle, avec une dimension de factor_num       

		self.embed_user_MLP = nn.Embedding(user_num, factor_num *(2**(num_layers - 1)))   # formule de progression géométrique
		self.embed_item_MLP = nn.Embedding(item_num, factor_num *(2**(num_layers - 1)))

		MLP_modules = []   # liste vide pour stocker les modules de la partie MLP
		for i in range(num_layers):
			input_size = factor_num*(2**(num_layers - i))  #taille d'entrée pour chaque couche
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, input_size//2)) #ajoute une couche linéaire fully connected à la liste, réduit la dimension de la représentation de moitié
			                              #entrée         sortie
			MLP_modules.append(nn.ReLU()) #Ajoute une fonction d'activation ReLU à la liste pour introduire de la non-linéarité
		self.MLP_layers = nn.Sequential(*MLP_modules) #crée une séquence de modules en utilisant la liste MLP_modules

		#Sans *, la liste entière serait traitée comme un seul argument. L'opérateur * assure que chaque module de la liste devient un argument distinct

		if self.model in ['MLP', 'GMF']:
			predict_size = factor_num 
		else:                                  #taille d'entrée de la couche de prédiction
			predict_size = factor_num*2
			
		self.predict_layer = nn.Linear(predict_size, 1)
#Crée une couche linéaire pour effectuer la prédiction finale.
#La taille d'entrée est predict_size et la sortie est une seule valeur, car c’est tâche de recommandation binaire (par exemple, prédire si un utilisateur va aimer un article)

		self._init_weight_()













		

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		if not self.model == 'NeuMF-pre':


             # Initialise les poids de l'embedding des utilisateurs/articles à partir d'une distribution normale avec une écart-type de 0.01. (des valeurs aléa)
			nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			for m in self.MLP_layers:
				if isinstance(m, nn.Linear): # si la couche actuelle (m) est une couche linéaire
					nn.init.xavier_uniform_(m.weight)    # Initialise les poids de la couche linéaire à l'aide de l'initialisation Xavier uniforme
					
			nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

		        # Initialise les poids de la couche de prédiction finale avec l'initialisation Kaiming uniforme, adaptée à la fonction d'activation sigmoid



			

			for m in self.modules():
				if isinstance(m, nn.Linear) and m.bias is not None:
					m.bias.data.zero_()

# Mettre les biais à zéro au départ aide le modèle à commencer l'apprentissage sans préférences initiales, afin d'ajuster les poids en fonction des données d'entraînement.
# le biais terme ajouté à la sortie d'une unité neuronale dans un réseau de neurones, permettant au modèle de mieux s'adapter aux données d'entraînement.
					
		else: #modèle NeuMF
		
			# embedding layers
			self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
			self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight) # Initialise les couches d'embedding avec les poids pré-entraînés des modèles GMF et MLP
			self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
			self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

			# mlp layers
			for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):     # copie de m2 vers m1 les poids et les biais
				if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
					m1.weight.data.copy_(m2.weight)       # ces 2 opérations sont spécifiques aux couches linéaires donc il faut vérifier que les couches sont linéaires
					m1.bias.data.copy_(m2.bias)

			# predict layers
			predict_weight = torch.cat([self.GMF_model.predict_layer.weight, self.MLP_model.predict_layer.weight], dim=1)
			# Concatène horizontalement les poids de la couche de prédiction des modèles GMF et MLP.

			
			predict_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias
			# Additionne les biais des couches de prédiction des modèles GMF et MLP.

			self.predict_layer.weight.data.copy_(0.5 * predict_weight) # Copie la moitié des poids concaténés vers la couche de prédiction du modèle   obtenir 100% de 200%
			self.predict_layer.bias.data.copy_(0.5 * predict_bias)     # Copie la moitié des biais additionés vers la couche de prédiction du modèle







#forward orchestre l'utilisation des composants du modèle définis dans le code antérieur pour effectuer une prédiction en fonction du type de modèle spécifié.


	def forward(self, user, item):
		if not self.model == 'MLP':
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			output_GMF = embed_user_GMF * embed_item_GMF
		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(user)
			embed_item_MLP = self.embed_item_MLP(item)
			output_MLP = self.MLP_layers(torch.cat((embed_user_MLP, embed_item_MLP), -1))

		if self.model == 'GMF':
			t = output_GMF
		elif self.model == 'MLP':
			t = output_MLP
		else:
			t = torch.cat((output_GMF, output_MLP), -1)   # -1 dimension à calculer

		prediction = self.predict_layer(t)
		return prediction.view(-1)            #view permet d'applatir le tensor



	    
