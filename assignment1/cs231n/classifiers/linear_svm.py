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
    # initialize the gradient as zero
    dW = np.zeros(W.shape)
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    margin_binary = np.zeros((num_train, num_classes))

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            # note delta = 1
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                margin_binary[i, j] = 1
                margin_binary[i, y[i]] -= 1

    dW = np.dot(margin_binary.T, X).T
    dW /= num_train
    dW += 2 * reg * W

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    loss += reg * np.sum(W * W)
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    delta = 1.0

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = np.dot(X, W)
    mask_y = np.eye(num_classes)[y].astype('bool')
    scores_y = scores[mask_y, np.newaxis]
    # compute the margins for all classes in one vector operation
    margins = scores - scores_y + delta
    # Ignore the y-th position and only consider margin on max wrong class
    margins[mask_y] = 0

    loss = np.sum(np.maximum(0, margins))
    loss /= num_train
    loss += reg * np.sum(W * W)

    # Calculate gradient
    margin_binary = np.copy(margins)
    margin_binary = (margin_binary > 0).astype('float')
    # Correct class
    margin_binary[mask_y] = -1 * np.sum(margin_binary, axis=1)

    dW = np.dot(margin_binary.T, X).T
    dW /= num_train
    dW += 2 * reg * W
    return loss, dW
