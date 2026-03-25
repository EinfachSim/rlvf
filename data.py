import numpy as np

class ValueProfileSampler:
    def __init__(self, noise=0.0):
        self.value_names = [
            "SD-Thought", "SD-Action", "Stimulation", "Hedonism",     
            "Achievement", "Power-Dom", "Power-Res", "Face",          
            "Sec-Personal", "Sec-Societal", "Tradition", "Conf-Rules",
            "Conf-Inter", "Humility", "Univ-Nature", "Univ-Concern",  
            "Univ-Tolerance", "Benev-Care", "Benev-Dep"
        ]
        self.n_values = len(self.value_names)
        angles = np.linspace(0, 2 * np.pi, self.n_values, endpoint=False)
        
        corr_matrix = np.zeros((self.n_values, self.n_values))
        for i in range(self.n_values):
            for j in range(self.n_values):
                corr_matrix[i,j] = np.cos(angles[i] - angles[j])
        
        random_noise = np.random.normal(0, noise, size=(self.n_values, self.n_values))
        noisy_corr = corr_matrix + (random_noise + random_noise.T) / 2 # Ensure symmetry
        np.fill_diagonal(noisy_corr, 1.0)
    
        # Ensure the matrix is positive semi-definite for sampling
        vals, vecs = np.linalg.eigh(noisy_corr)
        vals = np.maximum(vals, 1e-8)
        noisy_corr = vecs @ np.diag(vals) @ vecs.T
        self.corr = noisy_corr
    
    def sample_batch(self, num=1):
        profiles = np.random.multivariate_normal(np.zeros(self.n_values), self.corr, size=num)
    
        profiles = self.population_remap(profiles)

        row_means = profiles.mean(axis=1, keepdims=True)
        return profiles
    

    def remap_sigmoid(self, profiles, temp=5.0):
        return 1 + 5/(1+ np.exp(-profiles/temp))
    
    def population_remap(self, profiles):
        p1, p99 = np.percentile(profiles, [1, 99])
        # Rescale to [-1, 1] so sigmoid input is well-behaved
        rescaled = (profiles - p1)*5 / (p99 - p1) + 1

        rescaled = rescaled - 3.5
    
        # Clean up the 1% outliers so they don't break the Likert logic
        rescaled = np.clip(rescaled, -2.5, 2.5)

        return rescaled
