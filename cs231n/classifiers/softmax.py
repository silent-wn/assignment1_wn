import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train=X.shape[0]
  num_class=W.shape[1]
  scores=np.dot(X,W)
  scores_nom=np.zeros((num_train,num_class))
  for i in xrange(num_train):
  	scores[i,:]-=np.max(scores[i,:])

  scores_exp=np.exp(scores)
  for j in xrange(num_train):
  	scores_nom[j,:]=scores_exp[j,:]/(np.sum(scores_exp[j,:]))
    
  scores_nom_correct=scores_nom[np.arange(num_train),y]
  #scores_nom_correct=np.reshape(scores_nom_correct,(num_train,-1))
  counts=scores_exp/(scores_exp.sum(axis=1).reshape(num_train,1))
  counts[range(num_train),y] -= 1
  dW = np.dot(X.T,counts)
  loss=-np.log(scores_nom_correct)
  loss=np.sum(loss)/num_train
  loss+=0.5*reg*np.sum(W*W)
  dW=dW/num_train+reg *W
  


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss,dW



def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train=X.shape[0]
  num_class=W.shape[1]
  scores=np.dot(X,W)
  scores_nom=np.zeros((num_train,num_class))
  
  scores_max=np.amax(scores,axis=1)
  scores_max=np.reshape(scores_max,(num_train,-1))
  scores_exp=np.exp(scores)
  scores_nom=scores_exp/np.reshape((np.sum(scores_exp,axis=1)),(num_train,-1))
  scores_nom_correct=scores_nom[np.arange(num_train),y]
  
  counts=scores_exp/(scores_exp.sum(axis=1).reshape(num_train,1))
  counts[range(num_train),y] -= 1
  dW = np.dot(X.T,counts)

  loss=-np.log(scores_nom_correct)
  loss=np.sum(loss)/num_train
  loss+=0.5*reg*np.sum(W*W)
  dW=dW/num_train+reg *W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss,dW

