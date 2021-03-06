import numpy as np
from sklearn.metrics import f1_score
from sklearn import preprocessing
np_load_old = np.load
from sklearn import svm
from sklearn.metrics import accuracy_score
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



training_sentences = np.load('training_sentences.npy')
training_labels = np.load('training_labels.npy')
test_labels = np.load('test_labels.npy')
test_sentences = np.load('test_sentences.npy')

np.load = np_load_old




def normalize_data(train_data, test_data, type = None):
    if type == "STANDARD":
        scaler = preprocessing.StandardScaler()
    elif type == "min_max":
        scaler = preprocessing.MinMaxScaler()
    elif type == "l1":
        scaler = preprocessing.Normalizer(norm='l1')
    elif type == "l2":
        scaler =  preprocessing.Normalizer(norm='l2')
    scaler.fit(train_data)
    scaled_x_train = scaler.transform(train_data)
    scaled_x_test = scaler.transform(test_data)
    return(scaled_x_train,scaled_x_test)

class BagOfWords:
    def __init__(self):
        self.vocab = { }
        self.words = [ ]
        self.len = 0
    def build_vocabulary(self,data):
        for doc in data:
            for word  in doc:
                if word not in self.vocab:
                    self.vocab[word] = len(self.words)
                    self.words.append(word)

        self.len = len(self.words)
        self.words = np.array(self.words)
        #return self.len
    def get_features(self,data):
        feats = np.zeros((data.shape[0],self.len))
        for i in range(data.shape[0]):
            doc = data[i]
            for word in doc:
                if word in self.vocab:
                    feats[i][self.vocab[word]] =  feats[i][self.vocab[word]] + 1
        return feats

def coefficients(classifier, feature_names, top_features=10):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    print("-----Top Positive:-----")
    for i in range(10):
        print(feature_names[top_coefficients[i]])
    print("-----Top Negative:-----")
    for i in range(10,20):
        print(feature_names[top_coefficients[i]])

BOW = BagOfWords()
BOW.build_vocabulary(training_sentences)
train_feat = BOW.get_features(training_sentences)
test_feat = BOW.get_features(test_sentences)

norm_data = normalize_data(train_feat,test_feat,'l2')

obj = svm.SVC(1,kernel = 'linear')
obj.fit(norm_data[0],training_labels)
predictions = obj.predict(norm_data[1])
accur = accuracy_score(test_labels,predictions)
print("accuracy: ")
print( accur)
f1_scor = f1_score(test_labels,predictions,average =None)
print("f1_score: ")
print(f1_scor)

coefficients(obj, norm_data[1])


















