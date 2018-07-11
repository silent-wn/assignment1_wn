import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero  (3073,10)

  # compute the loss and the gradient
  num_classes = W.shape[1]#label num
  num_train = X.shape[0]#500 pictures(dev_set)
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)#1*3073.dot(3073*10)  scores scale:1*10
    correct_class_score = scores[y[i]]  #  D=500  y biaoshi 500zhang tupian meizhang de leibie 
    for j in xrange(num_classes): #j!=y[i]shi ,max(0,sj-syi+1)
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1,svm loss
      if margin > 0:
        loss += margin#all picture  sum of loss
        dW[:,j]+=X[i]
        dW[:,y[i]]-=X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train  #average

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  num_train=X.shape[0]
  num_classes=W.shape[1]
  scores=np.dot(X,W)
  scorescorrect=scores[np.arange(num_train),y]#500,meiyihang y label duiying de zhi tiquchulai
  scorescorrect=np.reshape(scorescorrect,(num_train,-1))#qudiao rongyu bianwei 500*1 shuzu np.reshape(a,(num1,num2))num2=-1 baochi yu num_train yizhi de hangshu 
  margin=scores-scorescorrect+1#500*10s
  margin=np.maximum(0,margin)
  margin[np.arange(num_train),y] = 0
  loss += np.sum(margin)/num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  margin[margin>0]=1
  row_sum=np.sum(margin,axis=1)
  margin[np.arange(num_train),y]=-row_sum
  dW+=np.dot(X.T,margin)/num_train+reg*W


  #counts = (margin > 0).astype(int)
  #counts[range(num_train), y] = - np.sum(counts, axis = 1)
  #dW += np.dot(X.T, counts) / num_train + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
