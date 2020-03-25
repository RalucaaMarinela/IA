#import pandas as pandas
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

import numpy as np
import re
import os
from collections import defaultdict
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import time

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
np.random.seed(42)
PRIMELE_N_CUVINTE = 1000


def accuracy(y, p):
    return 100 * (y==p).astype('int').mean()

def citeste_texte_din_director(fisier):
    date_text=[]
    iduri_text=[]

    with open(fisier,'r',encoding='utf-8') as fin:
        for line in fin:
            text_fara_punct = re.sub("[-.,;:!?\"\'\/()_*=`]", "", line)
            cuvinte_text = text_fara_punct.split()
            id_fis=cuvinte_text[0]
            iduri_text.append(id_fis)
            date_text.append(cuvinte_text[1:])
    return (iduri_text,date_text)

def functie_labels(fisier):
    labels_txt=[]
    with open(fisier,'r') as fin:
        for line in fin:
            text_fara_punct = re.sub("[-.,;:!?\"\'\/()_*=`]", "", line)
            cuvinte_text = text_fara_punct.split()
            labels_txt.append(cuvinte_text[1])
    return labels_txt
### citim datele ###
dir_path = 'D:\\ml\\train'
labels = np.loadtxt(os.path.join(dir_path, 'train_labels.txt'),usecols=1)

train_data_path = os.path.join(dir_path, 'train_samples.txt')
iduri_train, data = citeste_texte_din_director(train_data_path)

print(len(data))
print(data[0][:10])
### citim datele ###


### numaram cuvintele din toate documentele ###
contor_cuvinte = defaultdict(int)
for doc in data:
    for word in doc:
        contor_cuvinte[word] += 1

# transformam dictionarul in lista de tupluri ['cuvant1', frecventa1, 'cuvant2': frecventa2]
perechi_cuvinte_frecventa = list(contor_cuvinte.items())

# sortam descrescator lista de tupluri dupa frecventa
perechi_cuvinte_frecventa = sorted(perechi_cuvinte_frecventa, key=lambda kv: kv[1], reverse=True)

# extragem primele 1000 cele mai frecvente cuvinte din toate textele
perechi_cuvinte_frecventa = perechi_cuvinte_frecventa[0:PRIMELE_N_CUVINTE]

print ("Primele 10 cele mai frecvente cuvinte ", perechi_cuvinte_frecventa[0:10])


list_of_selected_words = []
for cuvant, frecventa in perechi_cuvinte_frecventa:
    list_of_selected_words.append(cuvant)
### numaram cuvintele din toate documentele ###


def get_bow(text, lista_de_cuvinte):
    '''
    returneaza BoW corespunzator unui text impartit in cuvinte
    in functie de lista de cuvinte selectate
    '''
    contor = dict()
    cuvinte = set(lista_de_cuvinte)
    for cuvant in cuvinte:
        contor[cuvant] = 0
    for cuvant in text:
        if cuvant in cuvinte:
            contor[cuvant] += 1
    return contor

def get_bow_pe_corpus(corpus, lista):
    '''
    returneaza BoW normalizat
    corespunzator pentru un intreg set de texte
    sub forma de matrice np.array
    '''
    bow = np.zeros((len(corpus), len(lista)))
    for idx, doc in enumerate(corpus):
        bow_dict = get_bow(doc, lista)
        ''' 
            bow e dictionar.
            bow.values() e un obiect de tipul dict_values 
            care contine valorile dictionarului
            trebuie convertit in lista apoi in numpy.array
        '''
        v = np.array(list(bow_dict.values()))

        bow[idx] = v
    return bow

data_bow = get_bow_pe_corpus(data, list_of_selected_words)
print ("Data bow are shape: ", data_bow.shape)
print("LEN DATA"+str(len(data)))
nr_exemple_train = 15000
nr_exemple_valid = 5000
nr_exemple_test = len(data) - (nr_exemple_train + nr_exemple_valid)

indici_train = np.arange(0, nr_exemple_train)
indici_valid = np.arange(nr_exemple_train, nr_exemple_train + nr_exemple_valid)
indici_test = np.arange(nr_exemple_train + nr_exemple_valid, len(data))



clf=LogisticRegression(C=100,solver='liblinear',random_state = 42)

clf.fit(data_bow[indici_train],labels[indici_train])

predictii = clf.predict(data_bow[indici_valid,:])
print ("Acuratete pe valid : ", accuracy(predictii, labels[indici_valid]))

indici_train_valid = np.concatenate([indici_train, indici_valid])
clf=LogisticRegression(C=100,solver='liblinear',random_state = 42)

clf.fit(data_bow[indici_train_valid, :], labels[indici_train_valid])
predictii = clf.predict(data_bow[indici_test])
print ("Acuratete pe test : ", accuracy(predictii, labels[indici_test]))

def scrie_fisier_submission(nume_fisier, predictii, iduri):
    with open(nume_fisier, 'w') as fout:
        fout.write("id,label\n")
        for id_text, pred in zip(iduri, predictii):
            fout.write(id_text + ',' + str(int(pred)) + '\n')
iduri_test,date_test=citeste_texte_din_director("D:\\ml\\test\\test_samples.txt") #folder kaggle date_test
print("Am citit ", len(iduri_test))
data_bow_test=get_bow_pe_corpus(date_test,list_of_selected_words)
clf=LogisticRegression(C=100,solver='liblinear',random_state = 42)
clf.fit(data_bow,labels)
predictii_test=clf.predict(data_bow_test)

scrie_fisier_submission('D:\\ml\\av57csv',predictii_test,iduri_test);
