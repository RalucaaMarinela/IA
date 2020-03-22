from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
# load training data
training_data = np.load('training_data.npy')
prices = np.load('prices.npy')
# print the first 4 samples
print('The first 4 samples are:\n ', training_data[:4])
print('The first 4 prices are:\n ', prices[:4])
# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)



def normalize_data(train_data, test_data):
    scaler = preprocessing.StandardScaler()
    if scaler != None:
        scaler.fit(train_data)
        scaled_x_train = scaler.transform(train_data)
        scaled_x_test = scaler.transform(test_data)
    return (scaled_x_train, scaled_x_test)

num_samples_fold = len(training_data) // 3
training_data_1, prices_1 = training_data[:num_samples_fold], prices[:num_samples_fold]
training_data_2, prices_2 = training_data[num_samples_fold: 2 * num_samples_fold], prices[num_samples_fold: 2 * num_samples_fold]
training_data_3, prices_3 = training_data[2 * num_samples_fold:], prices[2 * num_samples_fold:]

def step(train_data, train_labels, test_data, test_labels,model):
    normalized_train, normalized_test = normalize_data(train_data, test_data)
    mod = model.fit(normalized_train, train_labels)
    mae = mean_absolute_error(test_labels, mod.predict(normalized_test))
    mse = mean_squared_error(test_labels, mod.predict(normalized_test))
    return mae, mse

model = LinearRegression()

#Run 1
mae1, mse1 = step(np.concatenate((training_data_1, training_data_3)),       np.concatenate((prices_1, prices_3)),    training_data_2, prices_2,  model)
#Run 2
mae2, mse2 = step(np.concatenate((training_data_1,training_data_2)), np.concatenate((prices_1,prices_2)), training_data_3,prices_3,model)
#Run 3
mae3, mse3 = step(np.concatenate((training_data_2,training_data_3)),  np.concatenate((prices_2,prices_3)),   training_data_3,  prices_3,  model)
print("MAE")
print("Mae 1 : ", mae1)
print("Mae 2 : ", mae2)
print("Mae 3 : ", mae3)
print("MSE")
print("Mse 1 : ", mse1)
print("Mse 2 : ", mse2)
print("Mse 3 : ", mse3)

# 3

for alpha_ in [1, 10, 100, 1000]:
    model = Ridge(alpha=alpha_)

    print("alpha = %d -> \n" % alpha_)

    # Run 1
    mae1, mse1 = step(np.concatenate((training_data_1, training_data_3)),  np.concatenate((prices_1, prices_3)),  training_data_2,  prices_2,   model)
    # Run 2
    mae2, mse2 = step(np.concatenate((training_data_1, training_data_2)), np.concatenate((prices_1, prices_2)),    training_data_3,       prices_3,  model)
    # Run 3
    mae3, mse3 = step(np.concatenate((training_data_2, training_data_3)), np.concatenate((prices_2, prices_3)),   training_data_3, prices_3,  model)
    print("MAE")
    print("Mae 1 : ", mae1)
    print("Mae 2 : ", mae2)
    print("Mae 2 : ", mae2)
    print("MSE")
    print("Mse 1 : ", mse1)
    print("Mse 2 : ", mse2)
    print("Mse 3 : %f \n" % mse3)

#4
model = Ridge(10)
scaler = preprocessing.StandardScaler()
scaler.fit(training_data)
norm_train = scaler.transform(training_data)
model.fit(norm_train, prices)

print(model.coef_)
print(model.intercept_)

features = ["Anul fabricatiei",
            "Numarul de kilometri",
            "Mileage",
            "Motor",
            "Putere",
            "Numar de locuri",
            "Numar de locuitori",
            "Tip combustibil",
            "Tipul de transmitere"]

maxim = np.argmax(np.abs(model.coef_))
first_msf = features[int(maxim)]
second_msf = features[(maxim + 1)]
min_index = np.argmin(np.abs(model.coef_))
least_msf = features[int(min_index)]
print("Features are: ", features, "\n")
print("most significant feature: %s\n" % first_msf)
print("second most significant feature: %s\n" % second_msf)
print("least significant feature: %s\n" % least_msf)
