from VanillaNN import VanillaNN as Net
import VanillaNN
from sklearn.cross_validation import train_test_split
from sklearn import datasets
import numpy as np

# Artificial dataset - spirals
N = 5000  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
y = np.zeros(N * K, dtype='uint8')  # class labels
for j in xrange(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))/float(255)
y = digits.target

# num_points = 5000
# X = np.array([[random.getrandbits(1), random.getrandbits(1)] for i in range(num_points)])
# y = np.reshape([i ^ j for i, j in X], (num_points, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
nn = Net(64, 10, 3, [100, 75, 50], layer_type='TanH', reg_param=1e-9)
nn.train_network(X_train, y_train, alpha=0.5, epochs=20000, batch_size=50)
out = nn.test_network(X_train)
print "Training accuracy is {0} %".format(100*VanillaNN.evaluate_network(out, y_train))
out = nn.test_network(X_test)
print "Testing accuracy is {0} %".format(100*VanillaNN.evaluate_network(out, y_test))
