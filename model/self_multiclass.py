import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        for i in range(10):
            Y = (y == i).astype(int)
            classifier = svm.LinearSVC()
            classifier.fit(X,Y)
            binary_svm[i] = classifier
        return binary_svm

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        for i in range(9):
            for j in range(i+1,10):
                X_train_i = X[np.where(y==i)[0]]
                Y_train_i = y[np.where(y==i)[0]]
                X_train_j = X[np.where(y==j)[0]]
                Y_train_j = y[np.where(y==j)[0]]
                X_train = np.concatenate((X_train_i,X_train_j),axis=0)
                Y_train = np.concatenate((Y_train_i,Y_train_j),axis=0)
                Y_train = (Y_train == i).astype(int)
                classifier = svm.LinearSVC(random_state = 0)
                classifier.fit(X_train,Y_train)
                binary_svm[(i,j)] = classifier
        return binary_svm

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        N = X.shape[0]
        k = 10
        scores = np.zeros((N,k))
        for i in range(10):
            scores[:,i] = self.binary_svm[i].decision_function(X)
        return scores

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        N = X.shape[0]
        k = 10
        scores = np.zeros((N,k))
        for pair in self.binary_svm:
            i = pair[0]
            j = pair[1]
            clf = self.binary_svm[pair]
            pred = clf.predict(X)
            scores[:,i] += pred
            scores[:,j] += (pred==0).astype(int)
        return scores

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        K,d = W.shape
        N = X.shape[0]
        part1 = np.sum(np.square(np.linalg.norm(W,axis = 1)))*0.5
        middle = np.zeros(N)
        for i in range(N):
            tmp = np.zeros(K)
            for j in range(K):
                delta = 1 if j == y[i] else 0
                tmp[j] = 1 - delta + np.matmul(W[j],X[i])
            middle[i] = np.max(tmp) - np.matmul(W[y[i]], X[i])
        part2 = C*np.sum(middle)
        return part1+part2


    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        K,d = W.shape
        N = X.shape[0]
        grad = np.zeros((K,d))
        for i in range(N):
            tmp = np.zeros(K)
            for j in range(K):
                delta = 1 if j == y[i] else 0
                tmp[j] = 1 - delta + np.matmul(W[j],X[i])
            argmax_j = np.argmax(tmp)
            yi = y[i]
            if yi != argmax_j:
                grad[yi] -= C*X[i]
                grad[argmax_j] += C*X[i]
                


        grad += W
        return grad


        




