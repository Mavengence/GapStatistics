import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Callable
from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd


class GapStatistics:
    def __init__(self, 
                 algorithm: Callable = KMeans,
                 distance_metric: str = 'euclidean',
                 pca_sampling: bool = True,
                 return_params: bool = False):
        
        if isinstance(distance_metric, str):
            self.distance_metric = DistanceMetric.get_metric(distance_metric).pairwise
        elif callable(distance_metric):
            self.distance_metric = distance_metric
        else:
            raise TypeError("distance_metric must be either a string or a function")
        if not isinstance(pca_sampling, bool):
            raise TypeError('Please provide a bool for pca_sampling')
        if not isinstance(return_params, bool):
            raise TypeError('Please provide a bool for return_params')
        
        self.algorithm = algorithm
        self.pca_sampling = pca_sampling
        self.return_params = return_params
        
    def _calculate_Wks(self, K: int, X: list) -> list:    
        """
        Calculate the Wk's statistic for a given number of clusters (K) using K-means clustering.

        Parameters:
            K (int): The number of clusters (K) to evaluate the Wk's statistic.
            X (list): A list of data points (samples) to be used for clustering.

        Returns:
            list: A list containing the Wk's statistics for each number of clusters from 1 to K.
        """
        Wks = []

        for k in np.arange(1, K+1):        
            try:
                model = self.algorithm(n_clusters=k).fit(X)
                labels = model.predict(X)
                centroids = model.cluster_centers_
            except Exception as e:
                print('You have some problems with your customized provided clustering algorithm ', e)
            
            Ds = []
            
            for i in range(k):
                cluster_array = np.array(X[labels == i])
                # agnostic distance metric - Formula (1)
                d = self.distance_metric(cluster_array, centroids[i].reshape(1, -1))
                Ds.append(np.sum(d))

            pooled = 1 / (2 * len(X))
            # Formula (2)
            Wk = np.sum([D * pooled for D in Ds])  
            Wks.append(Wk)
        return Wks

    def _simulate_Wks(self, X: list, K: int, n_iterations: int) -> [list, list]:
        """
        Simulate the Wk's statistic for a given number of iterations using bootstrapping.

        Parameters:
            X (list): A list of data points (samples) to be used for bootstrapping.
            K (int): The number of clusters (K) to use in the calculation of Wk's statistic.
            n_iterations (int): The number of iterations to perform.

        Returns:
            np.ndarray: An array containing the Wk's statistics for each iteration.
                       The shape of the array will be (n_iterations,)
        """
        if self.pca_sampling:
            scaler = StandardScaler()
            scaled_X = scaler.fit_transform(X)
            _, _, VT = svd(scaled_X) # U, D, VT
            X_prime = X[0]@VT.T
        else:
            X_prime = X

        simulated_Wks = []

        for i in range(n_iterations):  
            Z_prime = np.random.uniform(np.min(X_prime), np.max(X_prime), size=(len(X), 2))

            if self.pca_sampling:
                sampled_X = Z_prime@VT
            else:
                sampled_X = Z_prime

            Wks_star = self._calculate_Wks(K=K, X=sampled_X)
            simulated_Wks.append(Wks_star)  

        sim_Wks = np.array(simulated_Wks)
        return sim_Wks

    def fit_predict(self, K: int, X: list, n_iterations: int = 20):
        """
        Perform gap statistics to find the optimal number of clusters (K) for a given dataset.

        Parameters:
            X (list): A list of data points (samples) to be used for clustering.
            K (int): The maximum number of clusters (K) to consider for finding the optimal K value.
            n_iterations (int): The number of iterations to perform for simulating Wk's statistics.

        Returns:
            int or tuple: If `return_params` is False, returns the optimal number of clusters (K).
                         If `return_params` is True, returns a tuple with the optimal K and additional
                         statistics used in gap analysis.
        """
        if not isinstance(K, int):
            raise TypeError('K must be of type int')
        if K > len(X):
            raise Exception('K must be smaller than the length of X')
        if not isinstance(n_iterations, int):
            raise TypeError('n_iterations must be of type int')
        if n_iterations > 500:
            raise Exception('n_iterations is too big, choose below 500')
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise TypeError('Please provide either a list or a numpy array for X')
        
        Wks = self._calculate_Wks(K=K, X=X)
        sim_Wks = self._simulate_Wks(K=K, X=X, n_iterations=n_iterations)    

        log_Wks = np.log(Wks)
        log_Wks_star = np.log(sim_Wks)

        sd_k = np.std(log_Wks_star, axis=0)
        sim_sks = np.sqrt(1 + (1 / n_iterations)) * sd_k

        gaps = np.mean(log_Wks_star - log_Wks, axis=0)

        optimum = 1  
        
        # THE GAP STATISTICS FORMULA
        for i in range(0, len(gaps) - 1):
            if(gaps[i] >= gaps[i+1] - sim_sks[i+1]):
                optimum = i
                break

        self.X = X
        self.n_iterations = n_iterations
        self.optimum = optimum
        self.params = {'Wks': Wks, 'sim_Wks': sim_Wks, 'sim_sks': sim_sks, 'gaps': gaps}

        if self.return_params:
            return optimum, self.params
        else:
            return optimum
        
    def plot(self, original_labels: list = None, colors: dict = None):
        def _clip_values(arr):
            """
                helper function
            """
            for i in range(len(arr)):
                if arr[i] < 0:
                    arr[i] = 0
                elif arr[i] > 1:
                    arr[i] = 1
            return arr
        
        # get variables
        Wks = self.params['Wks']
        sim_Wks = self.params['sim_Wks']
        sim_sks = self.params['sim_sks']
        gaps = self.params['gaps']

        log_Wks = np.log(Wks)
        log_sim_Wks = np.log(np.mean(sim_Wks, axis=0))

        # define colors
        if colors == None:
            colors = {0: "red", 1: "green", 2: 'blue', 3:'lightblue', 
            4:'yellow', 5:'pink', 6:'orange', 7:'purple', 8:'magenta', 9: 'black'}
        
        # train the clustering algorithm on the predicted optimum

        algorithm = self.algorithm(n_clusters=self.optimum).fit(self.X)
        labels = algorithm.predict(self.X)
        centers = algorithm.cluster_centers_
        
        plt.figure(figsize=(16,10))
    
        # matplotlib code
        if original_labels:
            mapped_colors = [colors[i] for i in original_labels]

        # Normalized variables
        log_Wks = log_Wks - np.max(log_Wks)
        log_sim_Wks = log_sim_Wks - np.max(log_sim_Wks, axis=0)
        Wks = Wks - np.max(Wks)


        # Top Left
        plt.subplot(2, 2, 1)      
        if original_labels: 
            for i, x in enumerate(self.X):
                plt.scatter(x[0], x[1], color=mapped_colors[i], cmap='rainbow')
            for j in range(len(centers)):
                plt.scatter(centers[j][0], centers[j][1], color="black", s=150)
        else:
            for i, x in enumerate(self.X):
                plt.scatter(x[0], x[1], color='black')

        if original_labels:
            plt.title("Original Data with {} Clusters".format(len(centers)))
        else:
            plt.title("Original Data")

        # Bottom right
        plt.subplot(2, 2, 4)  
        plt.plot(Wks, '-o', label="Wks from Data")
        plt.plot(log_Wks, '-o', label="Logged Wks from Data")
        plt.plot(log_sim_Wks, '-o', color="green", label=f"Logged Simulated Wks [{self.n_iterations}]")
        plt.title("Decrease of Within Cluster Distance")
        plt.legend()

        # Bottom left
        plt.subplot(2, 2, 3)       
        plt.plot(gaps, '-o', color='r')

        yminx = (((gaps - sim_sks) - np.min(gaps)) / (np.max(gaps) - np.min(gaps)))
        ymaxsx = (((gaps + sim_sks) - np.min(gaps)) / (np.max(gaps) - np.min(gaps)))

        clipped_yminx = _clip_values(yminx)
        clipped_ymaxsx = _clip_values(ymaxsx)

        for i in range(len(gaps)):
            plt.axvline(x=i, ymin=clipped_yminx[i], ymax=clipped_ymaxsx[i], color='black')
            
        plt.axvline(x=self.optimum, color='green')
        plt.title("Gap Statistics with optimum K at {}".format(self.optimum))

        # Top right    
        plt.subplot(2, 2, 2) 

        mapped_colors = [colors[i] for i in labels]

        for i, x in enumerate(self.X):
            plt.scatter(x[0], x[1], color=mapped_colors[i], cmap='rainbow')

        for j in range(len(centers)):
            plt.scatter(centers[j][0], centers[j][1], color="black", s=150)

        plt.title("Gap Statistics Data with {} Clusters".format(self.optimum))
        plt.show()