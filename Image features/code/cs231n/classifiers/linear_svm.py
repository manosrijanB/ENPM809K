from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

                #############################################################################
                # TODO:                                                                     #
                # Compute the gradient of the loss function and store it dW.                #
                # Rather than first computing the loss and then computing the derivative,   #
                # it may be simpler to compute the derivative at the same time that the     #
                # loss is being computed. As a result you may need to modify some of the    #
                # code above to compute the gradient.                                       #
                #############################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                
                # For incorrect classes, accumulate gradients from the input
                dW[:, j] += X[i]
                # For the correct class, accumulate negative gradients from the input
                dW[:, y[i]] -= X[i]
                
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #In this changed version, the gradient dW is computed along with the loss inside the loops, 
    #and the regularization term is added to both the loss and the gradient after exiting the loops
    # Average the loss and gradient over the number of training examples
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss and gradient
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    # Compute the scores (WX) for all training examples
    scores = X.dot(W)
    
    # Select the correct class scores
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    
    # Compute the margins matrix by subtracting the correct class scores from the scores matrix and adding 1 (delta).
    margins = np.maximum(0, scores - correct_class_scores + 1)
    
    # Set the margins for the correct classes to zero
    margins[np.arange(num_train), y] = 0
    
    # Compute the loss by summing all the positive margins and dividing by the number of training examples.
    loss = np.sum(margins) / num_train
    
    # Add regularization to the loss
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute the gradient
    # Set margins to either 0 or 1 for gradient calculation
    binary_margins = margins
    binary_margins[margins > 0] = 1
    
    # For each row in the binary margins matrix, set the value at the correct class to the negative sum of the row.
    row_sum = np.sum(binary_margins, axis=1)
    binary_margins[np.arange(num_train), y] = -row_sum
    
    # Compute the gradient by multiplying the transpose of the data matrix X with the binary margins matrix and dividing by the number of training examples.
    dW = X.T.dot(binary_margins) / num_train
    
    # Add regularization to the gradient
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
