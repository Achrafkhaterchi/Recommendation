
import os


# dataset 
dataset = 'ml-1m'








# model 
model = 'NeuMF-end'

assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']


# 'NeuMF-end' initialise les poids aléatoirement
# 'NeuMF-pre' initialise les poids à partir de modèles pré-entraînés (GMF et MLP) pour accélérer son apprentissage






# paths
main_path = 'C:/Users/sirin/Desktop/Etudes/Mini/mywork/Data/'

train_rating = main_path + '{}.train.rating'.format(dataset)          # {} prend le nom de la dataset
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'

# Vérification 
if not os.path.exists(train_rating) or not os.path.exists(test_rating) or not os.path.exists(test_negative):
    print(" fichiers manquants")
