# Auxiliary functions for svm.ipynb
import numpy as np


def kernel(x: np.ndarray, y: np.ndarray, type: str = 'linear', p: int = 2, sigma: float = 1.0) -> np.ndarray:
    """Computes kernel(x, y)

    Args:
        x (np.ndarray): first value
        y (np.ndarray): second value
        type (str, optional): Type of kernel to use. Defaults to 'linear'.
        p (int, optional): if type=="power", it is the power. Defaults to 2.
        sigma (float, optional): if type=="gaussian", it is sigma. Defaults to 1.0.

    Returns:
        np.ndarray: kernel(x, y)
    """
    if type == 'linear':
        return np.dot(x, y.T)
    if type == 'power':
        return (np.dot(x, y.T) + 1)**p
    if type == 'gaussian':
        return np.exp(np.linalg.norm(x-y)**2/(2*sigma**2))


def K_matrix(x :np.ndarray, type: str = 'linear', p: int = 2, sigma: float = 1.0) -> np.ndarray:

    """Computes the matrix K such as K[i, j] = kernel(x[i], x[j])


    Args:
        x (np.ndarray): input data
        type (str, optional): Type of kernel to use. Defaults to 'linear'.
        p (int, optional): if type=="power", it is the power. Defaults to 2.
        sigma (float, optional): if type=="gaussian", it is sigma. Defaults to 1.0.

    Returns:
        np.ndarray: K matrix
    """

    N = x.shape[0]
    K = np.zeros(shape=(N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if type == 'linear':
                K[i, j] = kernel(x[i], x[j], type='linear')
            if type == 'power':
                K[i, j] = kernel(x[i], x[j], type='power', p=p)
            if type == 'gaussian':
                K[i, j] = kernel(x[i], x[j], type='gaussian', sigma=sigma)
    return K


def objective(alpha: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Objective function of the SVM problem - it is to be minimized

    Args:
        alpha (np.ndarray): alpha vector of the weights
        t (np.ndarray): target vector
        K (np.ndarray): kernel matrix of the input vectors

    Returns:
        np.ndarray: the function to minimize
    """
    N = alpha.shape[0]
    right_sum = - alpha.sum()
    left_sum = 0
    for i in range(N):
        for j in range(N):
            left_sum += alpha[i] * alpha[j] * t[i] * t[j] * K[i, j]
    return 0.5 * left_sum + right_sum


def zerofun(alpha: np.ndarray, t: np.ndarray)-> np.ndarray:
    """Constraint function of the SVM problem to be fixed to 0

    Args:
        alpha (np.ndarray): alpha vector of the weights
        t (np.ndarray): target vector

    Returns:
        np.ndarray: 
    """
    return np.dot(alpha, t.T)


def b_fun(inputs: np.ndarray, alpha: np.ndarray, t: np.ndarray, idx: int = 0, type: str = 'linear', p: int = 2, sigma: float = 1.0) -> float:
    """_summary_

    Args:
        inputs (np.ndarray): array of the input data
        alpha (np.ndarray): alpha weight vector
        t (np.ndarray): t target vector
        idx (int, optional): idx index of the support vector to use to compute. Defaults to 0.
        type (str, optional): type of kernel to use. Defaults to 'linear'.
        p (int, optional): if kernel is power kernel, p is the power of it. Defaults to 2.
        sigma (float, optional): if gaussian kernel, controls the smoothness. Defaults to 1.0.

    Returns:
        float: the margin of the SVM classifier
    """
    value = 0.
    for id, x in enumerate(inputs):
        if type == 'linear':
            value += alpha[id] * t[id] * kernel(x=inputs[idx], y=x, type=type)
        if type == 'power':
            value += alpha[id] * t[id] * \
                kernel(x=inputs[idx], y=x, type=type, p=p)
        if type == 'gaussian':
            value += alpha[id] * t[id] * \
                kernel(x=inputs[idx], y=x, type=type, sigma=sigma)
    return float(value - t[idx])


def indicator(s: np.ndarray, alpha: np.ndarray, b: float, t: np.ndarray, inputs: np.ndarray, indexes: np.ndarray, type: str = 'linear', p: int = 2, sigma: float = 1.0) -> float:
    """Indicator function of the SVM problem : if >0 then t=-1, if <0 then t=1

    Args:
        s (np.ndarray): vector to be tested
        alpha (np.ndarray): weight vector
        b (float): margin of the SVM classifier
        t (np.ndarray): target vector
        inputs (np.ndarray): input data
        indexes (np.ndarray): indexes of the non zero alpha values in the input
        type (str, optional): Type of kernel to use. Defaults to 'linear'.
        p (int, optional): if type=="power", it is the power. Defaults to 2.
        sigma (float, optional): if type=="gaussian", it is sigma. Defaults to 1.0.


    Returns:
        float: Indicator function for the s vector
    """

    value = 0.
    for idx in indexes:
        if type == 'linear':
            value += alpha[idx] * t[idx] * kernel(s, inputs[idx], type=type, p=p, sigma=sigma)
        if type == 'power':
            value += alpha[idx] * t[idx] * \
                kernel(s, inputs[idx], type=type, p=p)
        if type == 'gaussian':
            value += alpha[idx] * t[idx] * \
                kernel(s, inputs[idx], type=type, sigma=sigma)
    return value - b
