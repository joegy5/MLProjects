import numpy as np

class Softmax():
    def forward(self, A):
        exp = np.exp(A)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, A, Y):
        return A - Y


class Flatten():
    def forward(self, A):
        return A.reshape(A.shape[0], -1)

    def backward(self, dZ, A):
        return dZ.reshape(A.shape)
        

class FFN():
    def __init__(self, in_features, out_features, beta1, beta2, learning_rate, epsilon=1e-7):
        self.beta1 = beta1
        self.beta2 = beta2 
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.W = np.random.normal(loc=0, scale=np.sqrt(2. / in_features), size=(out_features, in_features)) # He initialization
        self.b = np.zeros(shape=(1, out_features))
        self.VdW = np.zeros_like(self.W)
        self.Vdb = np.zeros_like(self.b)
        self.SdW = np.zeros_like(self.W)
        self.Sdb = np.zeros_like(self.b)

    def forward(self, A, use_relu=True):
        Z = np.matmul(A, self.W.T) + self.b
        if use_relu:
            return Z, np.maximum(0, Z)
        else:
            return Z

    def backward(self, A, dA, Z=None, used_relu=True):
        # Calculate weight and bias gradients
        if not used_relu: # next layer was softmax --> dZ is none
            dZ = dA
        else: # next layer was not softmax but relu instead --> dA is none
            dZ = dA * np.where(Z > 0, 1, 0)
        dW = (1. / A.shape[0]) * np.matmul(dZ.T, A)
        db = (1. / A.shape[0]) * np.sum(dZ, axis=0, keepdims=True)
        dAprev = np.matmul(dZ, self.W)
        # Update Adam gradients
        self.VdW = self.beta1 * self.VdW + (1. - self.beta1) * dW
        self.Vdb = self.beta1 * self.Vdb + (1. - self.beta1) * db
        self.SdW = self.beta2 * self.SdW + (1. - self.beta2) * (dW ** 2)
        self.Sdb = self.beta2 * self.Sdb + (1. - self.beta2) * (db ** 2)
        # Update weights and biases 
        self.W -= self.learning_rate * self.VdW / (np.sqrt(self.SdW) + self.epsilon)
        self.b -= self.learning_rate * self.Vdb / (np.sqrt(self.Sdb) + self.epsilon)
        return dAprev 

class Conv2D():
    def __init__(self, kernel_size, num_input_channels, num_output_channels, beta1, beta2, learning_rate, epsilon=1e-7):
        self.beta1 = beta1
        self.beta2 = beta2 
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.K = np.random.normal(
            loc=0, # mean 
            scale=np.sqrt(2. / (kernel_size * kernel_size * num_input_channels)), # variance (He initialization)
            size=(num_output_channels, kernel_size, kernel_size, num_input_channels))
        self.b = np.zeros((num_output_channels)) # one scalar bias value for each output filter
        self.VdK = np.zeros_like(self.K)
        self.Vdb = np.zeros_like(self.b)
        self.SdK = np.zeros_like(self.K)
        self.Sdb = np.zeros_like(self.b)

    def im2col(self, A):
        # implement im2col algorithm to efficiently implement convolution operation for all filters
        batch_size, in_dim, c_in, kernel_size = A.shape[0], A.shape[1], A.shape[-1], self.K.shape[1]
        out_dim = in_dim - kernel_size + 1
        im2col_strides = (A.strides[0], A.strides[1], A.strides[2], A.strides[1], A.strides[2], A.strides[3])
        im2col_shape = (batch_size, out_dim, out_dim, kernel_size, kernel_size, c_in)
        im2col_patches = np.lib.stride_tricks.as_strided(A, im2col_shape, im2col_strides).reshape(batch_size * out_dim * out_dim, -1)
        return im2col_patches

    def forward(self, A):
        batch_size, in_dim, c_out, kernel_size = A.shape[0], A.shape[1], self.K.shape[0], self.K.shape[1]
        out_dim = in_dim - kernel_size + 1
        im2col_patches = self.im2col(A)
        A_next = np.matmul(im2col_patches, self.K.reshape(c_out, -1).T).reshape(batch_size, out_dim, out_dim, c_out)
        Z = A_next + self.b[np.newaxis, np.newaxis, np.newaxis, :] # automatic broadcasting
        return Z, np.maximum(Z, 0)

    def backward(self, A, Z, dA):
        dZ = dA * np.where(Z > 0, 1, 0)
        batch_size, in_dim, out_dim, c_out, kernel_size, c_in = A.shape[0], A.shape[1], dZ.shape[1], self.K.shape[0], self.K.shape[1], self.K.shape[-1]
        pad = kernel_size - 1
        A_2d = self.im2col(A)
        dZ_2d = dZ.reshape(batch_size * out_dim * out_dim, c_out)
        dZ_padded_2d = self.im2col(np.pad(dZ, ((0, 0),(pad, pad),(pad, pad),(0, 0))))
        K_rotated_reshaped_2d = np.flip(self.K, axis=(1,2)).reshape(c_out * kernel_size * kernel_size, c_in)
        dAprev = np.matmul(dZ_padded_2d, K_rotated_reshaped_2d).reshape(batch_size, in_dim, in_dim, c_in)
        dK = (1. / batch_size) * np.matmul(dZ_2d.T, A_2d).reshape(c_out, kernel_size, kernel_size, c_in)
        db = (1./ batch_size) * dZ.sum(axis=(0,1,2))
        self.VdK = self.beta1 * self.VdK + (1. - self.beta1) * dK
        self.Vdb = self.beta1 * self.Vdb + (1. - self.beta1) * db
        self.SdK = self.beta2 * self.SdK + (1. - self.beta2) * (dK ** 2)
        self.Sdb = self.beta2 * self.Sdb + (1. - self.beta2) * (db ** 2)
        self.K -= self.learning_rate * self.VdK / (np.sqrt(self.SdK) + self.epsilon)
        self.b -= self.learning_rate * self.Vdb / (np.sqrt(self.Sdb) + self.epsilon)
        return dAprev


class Pooling2D():
    def __init__(self, filter_size, pooling_method):
        self.stride = filter_size
        self.filter_size = filter_size
        self.pooling_method = pooling_method

    def im2col(self, A):
        # implement im2col algorithm to efficiently implement convolution operation for all filters
        batch_size, in_dim, c_in = A.shape[0], A.shape[1], A.shape[-1]
        out_dim = (in_dim - self.filter_size) // self.filter_size  + 1
        # no overlapping windows, unlike in conv layer --> stride is filter size to move to next window (vertical or horizontal step)
        im2col_strides = (A.strides[0], A.strides[1] * self.stride, A.strides[2] * self.stride, A.strides[1], A.strides[2], A.strides[3])
        im2col_shape = (batch_size, out_dim, out_dim, self.filter_size, self.filter_size, c_in)
        return np.lib.stride_tricks.as_strided(A, im2col_shape, im2col_strides)
    
    def forward(self, A):
        im2col_patches = self.im2col(A)
        batch_size, out_dim, c_in = im2col_patches.shape[0], im2col_patches.shape[1], im2col_patches.shape[-1]
        im2col_patches = im2col_patches.reshape(batch_size, out_dim, out_dim, im2col_patches.shape[3] * im2col_patches.shape[4], c_in)
        if self.pooling_method == 'max':
            return np.max(im2col_patches, axis=3, keepdims=False)
        elif self.pooling_method == 'average':
            return np.mean(im2col_patches, axis=3, keepdims=False)
        
    def backward(self, dA, A, Anext=None):
        batch_size, in_dim, out_dim, c_in = A.shape[0], A.shape[1], dA.shape[1], dA.shape[-1]
        height_indices = np.arange(in_dim)
        height_indices = height_indices[height_indices // self.filter_size < out_dim][:, None]
        width_indices = np.arange(in_dim)
        width_indices = np.repeat(width_indices[None, width_indices // self.filter_size < out_dim], repeats=height_indices.shape[0], axis=0)
        dAprev = np.zeros(shape=(batch_size, in_dim, in_dim, c_in))
        dAprev[:, height_indices, width_indices, :] = dA[:, height_indices // self.filter_size, width_indices // self.filter_size, :]
        if self.pooling_method == 'max':
            Anext_expanded = np.zeros_like(dAprev)
            Anext_expanded[:, height_indices, width_indices, :] = Anext[:, height_indices // self.filter_size, width_indices // self.filter_size, :]
            return dAprev * ((Anext_expanded == A).astype(float))
        elif self.pooling_method == "average":
            return dAprev / (self.filter_size * self.filter_size)
