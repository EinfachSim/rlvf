import numpy as np

class ValueProfileSampler:
    def __init__(self):
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
        
        self.corr_matrix = corr_matrix*0.7 + np.eye(self.n_values) * 1e-8
    
    def sample_batch(self, num=1):
        return np.random.multivariate_normal(np.zeros(self.n_values), self.corr_matrix, size=num)
