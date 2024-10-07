import numpy as np
import math
import matplotlib.pyplot as plt

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        drop_class = iris[:, :-1]
        return np.mean(drop_class, axis=0)

    def empirical_covariance(self, iris):
        drop_class = iris[:, :-1]
        return np.cov(drop_class, rowvar=False)

    def feature_means_class_1(self, iris):
        filtration = iris[iris[:, -1] == 1]
        return self.feature_means(filtration)

    def empirical_covariance_class_1(self, iris):
        filtration = iris[iris[:, -1] == 1]
        return self.empirical_covariance(filtration)


class HardParzen:
    label_list = None
    label_inputs = None
    train_labels = None

    def __init__(self, h):
        self.h = h

    def fit(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
    def predict(self, test_data):
        
        predictions = []
        for test in test_data:
            distances = np.abs(self.train_inputs - test)
            # On remet les labels
            train_labels_column = self.train_labels[:, np.newaxis]
            distances = np.concatenate([distances, train_labels_column], axis=1)

            # On a donc un hypercube avec distance h centre sur le point du test data. Il faut donc drop toutes les rows ayant des attributs plus grands que h
            distances = distances[np.all(distances[:, :-1] <= self.h, axis=1)]  
            labels = distances[:, 4:]
            distances_somme =  np.sum(distances[:, :-1], axis=1).reshape(-1, 1)
            voisins = []

            # Finalement, on a un array avec [[distance, label],...] dans la fenetre de panzen.
            if distances_somme.size == 0:
                #Test data comme seed
                predictions.append(draw_rand_label(x=test,label_list=self.label_list))
            else:
                # On fait un count de chaque label
                for label in self.label_list:
                    labels_correspondant = (label == labels)
                    voisins.append(np.sum(labels_correspondant))
                predictions.append(np.argmax(np.array(voisins)) + 1)

        print(np.array(predictions))
        return np.array(predictions)

    


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def fit(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
    def predict(self, test_data):
        predictions = []
        for test in test_data:
            distances = np.abs(self.train_inputs - test)
            # On remet les labels
            train_labels_column = self.train_labels[:, np.newaxis]
            distances = np.concatenate([distances, train_labels_column], axis=1)
            distances_somme =  np.sum(distances[:, :-1], axis=1).reshape(-1, 1)
            # On applique le kernel et on prend le label associe a la plus grande valeure
            kernel = np.vectorize(lambda x: math.exp(-x / self.sigma**2))
            predictions.append(int(distances[np.argmax(kernel(distances_somme))][-1]))
        

        return np.array(predictions)


def split_dataset(iris):
    train_data = []
    validation_data = []
    test_data = []

    for i in range(iris.shape[0]):
        if i % 5 < 3:
            train_data.append(iris[i])
        elif i % 5 < 4:
            validation_data.append(iris[i])
        else:
            test_data.append(iris[i])

    return np.array(train_data), np.array(validation_data), np.array(test_data)

class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        hard = HardParzen(h)
        hard.fit(self.x_train , self.y_train)
        
        difference = hard.predict(self.y_val) - self.y_val

        correct = (difference == 0)
        rate = (self.y_val.size - np.sum(correct)) / self.y_val.size

        return float(rate)

    def soft_parzen(self, sigma):
        soft = SoftRBFParzen(sigma)
        soft.fit(self.x_train , self.y_train)
        
        difference = soft.predict(self.y_val) - self.y_val

        correct = (difference == 0)
        rate = (self.y_val.size - np.sum(correct)) / self.y_val.size
        return float(rate)


def make_graph(iris):
    train_data, validation_data, test_data = split_dataset(iris=iris)
    err = ErrorRate(train_data[:, :4], train_data[:, -1], validation_data[:, :4], validation_data[:, -1])
    x_axis = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    y_axis_soft = [err.soft_parzen(sigma) for sigma in x_axis]
    y_axis_hard = [err.hard_parzen(h) for h in x_axis]

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, y_axis_soft, marker='o', linestyle='-', color='b', label="SoftParzen(sigma)")
    plt.plot(x_axis, y_axis_hard, marker='o', linestyle='-', color='r', label="HardParzen(h)")

    plt.title("Erreur en utilisant Soft Parzen")
    plt.xlabel("sigma et h")
    plt.ylabel("Taux d'erreur")
    plt.legend()
    plt.show()

def get_test_errors(iris):
    train_data, validation_data, test_data = split_dataset(iris=iris)
    err = ErrorRate(train_data[:, :4], train_data[:, -1], validation_data[:, :4], validation_data[:, -1])

    x_axis = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    y_axis_soft = [err.soft_parzen(sigma) for sigma in x_axis]
    y_axis_hard = [err.hard_parzen(h) for h in x_axis]

    return [x_axis[np.argmin(y_axis_hard)], x_axis[np.argmin(y_axis_soft)]]

def random_projections(X, A):
    pass



def tests_q1():
    iris = np.genfromtxt('iris.txt')
    q1 = Q1()
    print("================================BEFORE================================")
    print(iris)
    print("================================MEANS================================")
    print(q1.feature_means(iris=iris))
    print("================================COVARIANCE================================")
    print(q1.empirical_covariance(iris=iris))
    print("================================MEANS1================================")
    print(q1.feature_means_class_1(iris=iris))
    print("================================COVARIANCE1================================")
    print(q1.empirical_covariance_class_1(iris=iris))
def tests_q2():
    iris = np.genfromtxt('iris.txt')
    q2 = HardParzen(0.5)
    print("================================PREDICT1================================")
    q2.fit(iris[:, :-1], iris[:, 4:])
    #6.200000 2.900000 4.300000 1.300000 2
    print(q2.predict(np.array([[6.200000, 2.900000, 4.300000, 1.300000],[6.200000, 2.900000, 4.300000, 2.300000]])))
def tests_q3():
    iris = np.genfromtxt('iris.txt')
    

    print("================================PREDICT1================================")
    print("================================HARD================================")
    q2 = HardParzen(0.5)
    print(iris[:, :-1])
    q2.fit(iris[:, :-1], iris[:, 4:].reshape(1, -1)[0])
    print(q2.predict(np.array([[6.200000, 2.900000, 4.300000, 1.300000],[6.200000, 2.900000, 4.300000, 2.300000]])))
    print("================================SOFT================================")
    q3 = SoftRBFParzen(1)
    q3.fit(iris[:, :-1], iris[:, 4:].reshape(1, -1)[0])
    print(q3.predict(np.array([[6.200000, 2.900000, 4.300000, 1.300000],[6.200000, 2.900000, 4.300000, 2.300000]])))

def test_q4():
    iris = np.genfromtxt('iris.txt')

    train_data, validation_data, test_data = split_dataset(iris=iris)
    err = ErrorRate(train_data[:, :4], train_data[:, -1], validation_data[:, :4], validation_data[:, -1])
    print(err.hard_parzen(0.5))
    #make_graph(iris)
    #print(get_test_errors(iris))




if __name__ == "__main__":
   #tests_q1()
   #tests_q2()
   #tests_q3()
   test_q4()
