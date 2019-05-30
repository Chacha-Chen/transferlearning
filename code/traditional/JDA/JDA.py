# encoding=utf-8
"""
    Created on 21:29 2018/11/12 
    @author: Jindong Wang
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
# from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from time import time
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.decomposition import PCA

WHERTHER_PCA = True

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))
        # Ys = np.array(Ys)

        M = e * e.T * C
        Y_tar_pseudo = None
        for t in range(self.T):
            N = 0
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / len(np.array(Ys)[np.where(np.array(Ys) == c)])
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M += N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            global names
            names = [
                "Nearest Neighbors",
                #      "Linear SVM",
                #      "RBF SVM",
                     # "Gaussian Process",
                     # "Decision Tree",
                     # "Random Forest",
                     # "Extra Tree",
                     # "Neural Net",
                     # "AdaBoost",
                     # "Naive Bayes",
                     # "QDA"
                     ]

            classifiers = [
                KNeighborsClassifier(1),
                # SVC(kernel="linear", C=2.5),
                # SVC(gamma=2, C=2.5),
                # GaussianProcessClassifier(1.0 * RBF(1.0)),
                # DecisionTreeClassifier(max_depth=5),
                # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                # ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0),
                # MLPClassifier(alpha=1, max_iter=2000),
            ]

            accs = []
            for name, clf in zip(names, classifiers):
                t0 = time()
                print('begin %s fit' % name)
                clf.fit(Xs_new, Ys.ravel())
                # print('clf fit done')
                Y_tar_pseudo = clf.predict(Xt_new)
                acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
                accs.append(sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo))
                list_acc.append(acc)
                print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, acc))
                print("clf done in %0.3fs" % (time() - t0))
            # return acc, y_pred



            # clf = KNeighborsClassifier(n_neighbors=1)
            # clf.fit(Xs_new, Ys.ravel())
            # Y_tar_pseudo = clf.predict(Xt_new)
            # acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            # list_acc.append(acc)
            # print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, acc))
        return accs, Y_tar_pseudo, list_acc


if __name__ == '__main__':
    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    domains = ['Art_Art.csv',"Clipart_Clipart.csv","Product_Product.csv"]
    t_domains = ["Art_RealWorld.csv","Clipart_RealWorld.csv","Product_RealWorld.csv"]
    datapath = "../data/Office-Home_resnet50/"
    for i in range(len(domains)):
        for j in range((len(domains))):
            if i == j:
                print("source:",domains[i])
                print("target",t_domains[j])
                src, tar = datapath + domains[i], datapath + t_domains[j]
                # src, tar = '/Users/chenchacha/transferlearning/code/traditional/data/' + domains[i], '/Users/chenchacha/transferlearning/code/traditional/data/' + domains[j]
                # src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Source = pd.read_csv(src,header=None)
                Target = pd.read_csv(tar,header=None)
                Ys = Source[2048]
                Xs = Source.iloc[:, 0:2048]
                Yt = Target[2048]
                Xt = Target.iloc[:, 0:2048]
                if WHERTHER_PCA:
                    t0=time()
                    pca = PCA(n_components=0.95,svd_solver='full').fit(Xs)
                    # print("done in %0.3fs" % (time() - t0))
                    # t0 = time()
                    Xs = pca.transform(Xs)
                    Xt = pca.transform(Xt)
                    print(Xs.shape,Xs.shape)
                # Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                jda = JDA(kernel_type='primal', dim=30, lamb=1, gamma=1)
                accs, ypre, list_acc = jda.fit_predict(Xs, Ys, Xt, Yt)
                # print(acc)
                for name,acc in zip(names,accs):
                    print(name,acc)

