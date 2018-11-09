#include "adhoc/llo/api.hpp"

/// sigmoid function: f(x) = 1/(1+e^-x)
llo::DataNode sigmoid (llo::DataNode x);

/// tanh function: f(x) = (e^(2*x)+1)/(e^(2*x)-1)
llo::DataNode tanh (llo::DataNode x);

/// softmax function: f(x) = e^x / sum(e^x)
llo::DataNode softmax (llo::DataNode x);
