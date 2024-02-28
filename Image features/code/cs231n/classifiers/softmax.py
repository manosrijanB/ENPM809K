from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    #Loop over each training sample in the mini-batch
    for i in range(num_train):
        #Compute the raw scores (logits) 
        scores = X[i].dot(W)
        scores -= np.max(scores)  # For numerical stability, subtract the maximum score from each score. This prevents large exponentials and possible overflow.
        
        # Compute softmax loss
        sum_exp_scores = np.sum(np.exp(scores))
        correct_exp_score = np.exp(scores[y[i]])
        loss += -np.log(correct_exp_score / sum_exp_scores)
        
        # Compute gradient 
        #Loop over each class and compute the gradient for each class and also Update the gradient matrix dW based on whether the current class is the correct class or not.
        for j in range(num_classes):
            softmax_output = np.exp(scores[j]) / sum_exp_scores
            if j == y[i]:
                dW[:, j] += (softmax_output - 1) * X[i].T
            else:
                dW[:, j] += softmax_output * X[i].T
                
    # Average over the number of training examples and add regularization
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    
    # Compute the scores (logits) using vectorized operations
    scores = X.dot(W)
    
    # For numerical stability, subtract the max score from each score
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # Compute the softmax scores
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    #Normalize the exponentiated scores by dividing each score by the sum of scores for that training example
    softmax_scores = exp_scores / sum_exp_scores
    
    # Compute the softmax loss
    correct_class_scores = softmax_scores[np.arange(num_train), y]
    #Average the loss by dividing by the number of training examples and add the regularization term to the loss.
    loss = -np.sum(np.log(correct_class_scores)) / num_train
    loss += reg * np.sum(W * W)
    
    # Compute the gradient
    softmax_scores[np.arange(num_train), y] -= 1
    #Compute the gradient dW using a vectorized dot product between the transposed input features X.T and the adjusted softmax_scores.
    #Average the gradient by dividing by the number of training examples.
    dW = X.T.dot(softmax_scores) / num_train
   # Add the regularization term to the gradient.
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
