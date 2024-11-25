Feed Forward Neural Network built from scratch (only Numpy, as well as scikit_learn for preprocessing) to predict if loan will be defaulted
Includes Adam Optimizer and Batchnorm
Architecture: 
  - Layer 1: 220 RELU neurons (with Batch Normalization after)
  - Layer 2: 100 RELU neurons (with Batch Normalization after)
  - Layer 3: 50 RELU neurons (with Batch Normalization after)
  - Layer 4: 30 RELU neurons (with Batch Normalization after)
  - Layer 5: 1 SIGMOID neuron (NO Batch Normalization after, as this is the output layer)
