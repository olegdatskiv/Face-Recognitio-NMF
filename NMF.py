import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


class NMFDecomposition:
    def __init__(self, k, max_iter=1000, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def nmf_fit(self, X):
        """
        Non-negative Matrix Factorization (NMF) algorithm.

        Parameters:
        X: numpy array of shape (n_samples, n_features)
           Input data matrix.
        k: int
           Number of components to extract.
        max_iter: int, optional (default=100)
           Maximum number of iterations.

        Returns:
        W: numpy array of shape (n_samples, k)
           Matrix of basis vectors.
        H: numpy array of shape (k, n_features)
           Matrix of coefficients.
        """

        # Initialize W and H matrices randomly
        n_samples, n_features = X.shape
        W = np.random.rand(n_samples, self.k)
        H = np.random.rand(self.k, n_features)

        # Perform NMF using multiplicative updates
        for i in range(self.max_iter):
            # Update H
            numerator = np.dot(W.T, X)
            denominator = np.dot(np.dot(W.T, W), H) + 1e-9
            H *= numerator / denominator

            # Update W
            numerator = np.dot(X, H.T)
            denominator = np.dot(W, np.dot(H, H.T)) + 1e-9
            W *= numerator / denominator

            # Compute the error
            error = np.sqrt(np.mean((X - np.dot(W, H)) ** 2))

            # Check for convergence
            if error < self.tol:
                break

        return W, H


if __name__ == "__main__":
    # Load the image
    img = plt.imread('dataset/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')

    # Convert the image to a 2D matrix
    X = img.reshape(-1, img.shape[-1])

    # Set the number of components
    k = 10

    nmf = NMFDecomposition(k)

    W, H = nmf.nmf_fit(X)

    # Reconstruct the image using the learned basis and coefficients
    X_hat = np.dot(W, H)
    img_hat = X_hat.reshape(img.shape)

    # Display the original and reconstructed images
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[1].imshow(img_hat)
    ax[1].set_title('Reconstructed Image')
    plt.show()

    model = NMF(n_components=k, init='random', random_state=0)
    W = model.fit_transform(X)
    H = model.components_

    # Reconstruct the image using the learned basis and coefficients
    X_hat = np.dot(W, H)
    img_hat = X_hat.reshape(img.shape)

    # Display the original and reconstructed images
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[1].imshow(img_hat)
    ax[1].set_title('Reconstructed Image by sklearn')
    plt.show()
