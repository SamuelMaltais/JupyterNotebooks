import numpy as np

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
    def __init__(self, h):
        self.h = h

    def fit(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        pass

    def predict(self, test_data):
        pass


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def fit(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        pass

    def predict(self, test_data):
        pass


def split_dataset(iris):
    pass


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        pass

    def soft_parzen(self, sigma):
        pass


def get_test_errors(iris):
    pass


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



if __name__ == "__main__":
    tests_q1()
