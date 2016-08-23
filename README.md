# Ladder
A simple neural network library for python. Currently supports simple feedforward neural nets. 
Below is a sample code demonstrating its use.

```python
from VanillaNN import VanillaNN as Net

# Some code here to fetch training data
# and split into training and test sets

nn = Net(input_dimension, num_classes, num_hidden_layers,
          [100, 75, 50], layer_type='TanH', reg_param=1e-9)
nn.train_network(X_train, y_train, alpha=0.5, epochs=20000, batch_size=50)
out = nn.test_network(X_train)
print "Training accuracy is {0} %".format(100*VanillaNN.evaluate_network(out, y_train))
out = nn.test_network(X_test)
print "Testing accuracy is {0} %".format(100*VanillaNN.evaluate_network(out, y_test))
```
