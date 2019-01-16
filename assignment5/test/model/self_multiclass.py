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
        y_uniq = np.unique(y)
        
        for label in y_uniq:
            y_temp = list(y)
            for index in range(len(y_temp)):
                if y_temp[index] != label:
                    y_temp[index] = 0
                else:
                    y_temp[index] = 1
            clf = svm.LinearSVC(random_state=12345)
            clf.fit(X, y_temp)
            binary_svm[label] = clf
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
        # print("this is the type of y: ", y.shape[0])
        
        y_uniq = np.unique(y)
        for i in y_uniq:
            for j in y_uniq:
                if j > i:
                    X_temp = []
                    y_temp = []
                    label_pair = (i,j)
                    for y_index in range(y.shape[0]):
                        if y[y_index] == label_pair[0] or y[y_index] == label_pair[1]:
                            X_temp.append(X[y_index])
                            y_temp.append(y[y_index])
                    clf = svm.LinearSVC(random_state=12345)
                    clf.fit(X_temp, y_temp)
                    binary_svm[label_pair] = clf
        return binary_svm


    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = []

        for label in self.labels:
            score = self.binary_svm[label].decision_function(X)
            scores.append(score)
        scores = np.array(scores).T

        return scores

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
 


        scores = []
        scores_pre = []
        for k, v in self.binary_svm.items():
            score = v.predict(X)
            scores_pre.append(score)
        scores_pre = np.array(scores_pre).T

        for line in scores_pre:
            counts = [0] * len(self.labels)
            uniq_lab, uniq_counts = np.unique(line, return_counts=True)
            for index in range(len(uniq_lab)):
                counts[uniq_lab[index]] = uniq_counts[index]
            scores.append(counts)
        scores = np.array(scores)
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
        N = X.shape[0]
        K = W.shape[0]
        W_square =  np.linalg.norm(W, axis=1)
        W_square = np.power(W_square, 2.0)
        loss_helper1 = 0.5 * np.sum(W_square)


        
        sum_helper = 0
        for i in range(N):
            max_helper = []
            for j in range(K):                
                if y[i] == j:
                    sign = 1
                else:
                    sign = 0
                max_helper.append(1 - sign + np.dot(W[j], X[i].T)- np.dot(W[y[i]], X[i].T))

            max_v = max(max_helper) 
            sum_helper = sum_helper + max_v


        loss_helper2 = C * sum_helper

        return loss_helper1 + loss_helper2

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
        N = X.shape[0]
        K = W.shape[0]
        grad = W.copy()

        for i in range(N):
            max_helper = []
            for j in range(K):
                if y[i] == j:
                    sign = 1
                else:
                    sign = 0
                max_helper.append(1 - sign + np.dot(W[j], X[i].T)- np.dot(W[y[i]], X[i].T))
            max_index = max_helper.index(max(max_helper))
            grad[max_index] = grad[max_index] + C * X[i]
            grad[y[i]] = grad[y[i]] - C * X[i]

        return grad

