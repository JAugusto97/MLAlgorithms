import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# KNN is probably the simplest classification algorithm. It's training consists on memorizing the training set.
# It classifies a new example xi by finding the classes of the k nearest examples to xi and deciding xi's class
# by finding the majority of the neighbors classes.
# The parameter k can either overfit or underfit the model. If k is too low (ex: k==1), then the model can overfit
# If k is too large(ex: k==n), then the model can underfit.
# Note: k==n will aways predict the majority class on the dataset.

class KNN():
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        """Memorizes the training examples"""
        self.X = X
        self.y = y

    def distance(self, p1, pn):
        """Returns the index of the k closest points to p1"""
        distances = list()
        for point in pn:
            distances.append(np.sqrt(np.sum((p1-point)**2)))

        return (np.argpartition(distances, self.k)[:self.k])

    def vote(self, pk, y):
        """The class of a point is equal to the class of the majority of it's k neighbors"""
        votes = {}
        for point in pk:
            if y[point] in votes.keys():
                votes[y[point]] += 1
            else:
                votes[y[point]] = 1

        return max(votes, key=votes.get)

    def predict(self, x):
        y_pred = list()
        for xi in x:
            knn = self.distance(xi, self.X)
            target = self.vote(knn, self.y)
            y_pred.append(target)

        return np.array(y_pred)

    def score(self, y_true, y_pred):
        count = 0
        m = len(y_true)
        for i in range(m):
            if y_true[i] == y_pred[i]:
                count += 1

        return count/m

# Loading dataset
iris = load_iris()

X = iris.data
y = iris.target

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

# Running KNN with k==3
clf = KNN(k=3)
clf.fit(X_train, y_train)

# Scoring with accuracy and confusion matrix
print("Accuracy: {:.2f}".format(clf.score(y_test, clf.predict(X_test))))
print("Confusion Matrix:\n", confusion_matrix(y_test,clf.predict(X_test)))

