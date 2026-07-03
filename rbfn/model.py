import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

class RBFN:
    def __init__(self, num_rbf_neurons):
        self.num_rbf_neurons = num_rbf_neurons
        self.centers = None
        self.widths = None
        self.weights = None

    def _calculate_widths(self, centers):
        # A common heuristic: set width based on the maximum distance between centers
        # divided by sqrt(2 * num_rbf_neurons)
        if centers.shape[0] == 1:
            # If only one center, width cannot be calculated from distances between centers
            # A fallback, possibly arbitrary value or user-defined default
            return np.array([1.0]) # Or some other sensible default

        dmax = np.max(pdist(centers))
        sigma = dmax / np.sqrt(2 * self.num_rbf_neurons)
        return np.full(self.num_rbf_neurons, sigma)

    def _rbf_activations(self, X):
        # Calculate Euclidean distance between X and each center
        # Then apply a Gaussian kernel function
        # D: (n_samples, num_rbf_neurons)
        D = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        # Phi: (n_samples, num_rbf_neurons)
        phi = np.exp(-(D**2) / (2 * self.widths**2))
        return phi

    def fit(self, X_train, y_train):
        # 1. Use K-Means to find RBF centers
        kmeans = KMeans(n_clusters=self.num_rbf_neurons, random_state=42, n_init=10)
        kmeans.fit(X_train)
        self.centers = kmeans.cluster_centers_

        # 2. Calculate the widths of the RBF neurons
        self.widths = self._calculate_widths(self.centers)

        # 3. Compute the RBF activation matrix (phi matrix) for X_train
        phi_train = self._rbf_activations(X_train)

        # 4. Solve for the output layer weights using a linear solver
        # Add a bias term to the phi matrix by adding a column of ones
        phi_train_b = np.hstack([np.ones((phi_train.shape[0], 1)), phi_train])
        self.weights = np.linalg.lstsq(phi_train_b, y_train, rcond=None)[0]

    def predict(self, X):
        # Compute the RBF activation matrix for X
        phi_X = self._rbf_activations(X)
        # Add a bias term
        phi_X_b = np.hstack([np.ones((phi_X.shape[0], 1)), phi_X])
        # Multiply by the trained output layer weights
        predictions = np.dot(phi_X_b, self.weights)
        return predictions

print("RBFN class defined successfully.")