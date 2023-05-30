from builtins import object
import numpy as np

from docs.layers import *
from docs.fast_layers import *
from docs.layer_utils import *

class ConvolutionalNeuralNet(object):
    """
    A convolutional neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a editable softmax loss function. This will also implement
    spatial batch normalization. For a network with L layers,
    the architecture will be

    {conv - spatial batch norm - relu - 2x2 max pool} x (L - 2) - affine - relu - affine - editable softmax

    The {...} block is repeated L - 1 times.
    """
    
    def __init__(self, layers = 3, input_dim=(3, 32, 32), num_filters=[32], filter_sizes=[7],
                 hidden_dim=40, strides = [1], paddings = [0], pools = [2], temperature = 0, 
                 num_classes=100, weight_scale=1e-3, reg=0.0, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Array of number of filters to use in the convolutional layers
        - filter_sizes: Array of sizes of filters to use in the convolutional layers
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - strides : Array of strides to use in the convolutional layers
        - paddings : Array of pad to use in the convolutional layers
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        
        self.params = {}
        self.strides = strides
        self.paddings = paddings
        self.reg = reg
        self.dtype = dtype
        self.num_layers = layers #at-least 3
        self.use_gumbel = (temperature > 0)
        self.t = temperature
        
        C, H, W = input_dim
        
        for L in range(self.num_layers - 2):
            
            if(L == 0):
                self.params["W" + str(L + 1)] = weight_scale*np.random.randn(num_filters[L], C, 
                            filter_sizes[L], filter_sizes[L])
                
            else:
                self.params["W" + str(L + 1)] = weight_scale*np.random.randn(num_filters[L], 
                            num_filters[L-1], filter_sizes[L], filter_sizes[L])
                
                
            H = 1 + int((H - filter_sizes[L] + 2*paddings[L])/strides[L])
            H /= 2
            self.params["b" + str(L + 1)] = np.zeros((num_filters[L]))
            self.params["gamma" + str(L + 1)] = np.ones((num_filters[L]))
            self.params["beta" + str(L + 1)] = np.zeros((num_filters[L]))
            
        
        #H = (H + 2*paddings[-1] - filter_sizes[-1])/strides[-1]
        self.params["W" + str(self.num_layers - 1)] = weight_scale*np.random.randn(num_filters[-1]*H*H, hidden_dim)
        
        self.params["b" + str(self.num_layers - 1)] = np.zeros((1, hidden_dim))
        
        self.params["W" + str(self.num_layers)] = weight_scale*np.random.randn(hidden_dim, num_classes)
        
        self.params["b" + str(self.num_layers)] = np.zeros((1, num_classes))
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            
        
        self.bn_params = []
        self.bn_params = [{'mode': 'train',
                           'eps' : 1e-5,
                           'momentum' : 0.9
                          } for i in range(self.num_layers - 2)]
        
        self.pool_params = []
        self.pool_params = [{'pool_height': pools[i],
                             'pool_width': pools[i],
                             'stride' : 2
                             } for i in range(self.num_layers - 2)] 
            
        
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        
        """
        
        cache = {}
        for L in range(self.num_layers - 2):
            
            W = self.params["W" + str(L + 1)]
            b = self.params["b" + str(L + 1)]
            gamma = self.params["gamma" + str(L + 1)]
            beta = self.params["beta" + str(L + 1)]
            
            conv_param = {"stride" : self.strides[L],
                          "pad" : self.paddings[L]}
            
            if(L == 0):
                scores, cache["L" + str(L + 1)] = conv_bn_relu_maxpool_forward(X, W, b, gamma, beta,
                              conv_param,
                              self.bn_params[L],
                              self.pool_params[L])   
            else:
                scores, cache["L" + str(L + 1)] = conv_bn_relu_maxpool_forward(scores, W, b, gamma, beta,
                              conv_param,
                              self.bn_params[L],
                              self.pool_params[L])
        
        scores, cache["L" + str(self.num_layers - 1)] = affine_relu_forward(scores, 
                      self.params["W" + str(self.num_layers - 1)],
                      self.params["b" + str(self.num_layers - 1)])
        
        scores, cache["L" + str(self.num_layers)] = affine_forward(scores, 
                      self.params["W" + str(self.num_layers)], 
                      self.params["b" + str(self.num_layers)])

        if y is None:
            return scores

        loss, grads = 0, {}
        if(self.use_gumbel):
            loss, dA_final = gumbel_softmax_loss(scores, y, self.t, 1e-8)
        else:
            loss, dA_final = softmax_loss(scores, y)
        
        loss += self.reg*0.5*(np.sum(self.params["W" + str(self.num_layers)]*self.params["W" + str(self.num_layers)]))
        loss += self.reg*0.5*(np.sum(self.params["W" + str(self.num_layers - 1)]*self.params["W" + str(self.num_layers - 1)]))
        
        dA_prev, grads["W" + str(self.num_layers)], grads["b" + str(self.num_layers)] = affine_backward(dA_final, cache["L" +  str(self.num_layers)])
        
        grads["W" + str(self.num_layers)] += self.reg*self.params["W" + str(self.num_layers)]
        
        dA_prev, grads["W" + str(self.num_layers - 1)], grads["b" + str(self.num_layers - 1)] = affine_relu_backward(dA_prev, cache["L" + str(self.num_layers - 1)])
        
        grads["W" + str(self.num_layers - 1)] += self.reg*self.params["W" + str(self.num_layers - 1)]
        
        for L in range(self.num_layers-3, -1, -1):
            
            W = self.params["W" + str(L + 1)]
            b = self.params["b" + str(L + 1)]
            gamma = self.params["gamma" + str(L + 1)]
            beta = self.params["beta" + str(L + 1)]
            
        
            loss += self.reg*0.5*(np.sum(W*W))
            
            dA_prev, grads["W" + str(L + 1)], grads["b" + str(L + 1)], grads["gamma" + str(L + 1)], grads["beta" + str(L + 1)] = conv_bn_relu_maxpool_backward(dA_prev, cache["L" +  str(L + 1)])
        
            grads["W" + str(L + 1)] += self.reg*self.params["W" + str(L + 1)]
            
        return loss, grads
    
    
##HELPER FUNCTIONS

def conv_bn_relu_maxpool_forward(X, W, b, gamma, beta, conv_param, bn_param, pool_param):
    
    conv, conv_cache = conv_forward_fast(X, W, b, conv_param)
    
    conv_hat, bn_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
    
    conv_relu, relu_cache = relu_forward(conv_hat)
    
    out, pool_cache = max_pool_forward_fast(conv_relu, pool_param)
    
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    
    return out, cache

def conv_bn_relu_maxpool_backward(dout, cache):
    
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    
    dA_pool = max_pool_backward_fast(dout, pool_cache)
    
    dA_hat = relu_backward(dA_pool, relu_cache)
    
    dA_norm, dgamma, dbeta = spatial_batchnorm_backward(dA_hat, bn_cache)
    
    dA, dW, db = conv_backward_fast(dA_norm, conv_cache)
    
    return dA, dW, db, dgamma, dbeta
    