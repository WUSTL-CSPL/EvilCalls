import numpy as np
import numpy.linalg as nlg

class StormOptimizer():    
    # lr-->learning rate
    # c-->parameter to be swept over logarithmically spaced grid as per paper
    # w and k to be set as 0.1 as per paper

    def __init__(self,c=100):
        self.gradient = []
        self.momentum = 0
        self.sqrgradnorm = 0
        self.c = c
        self.count = 0
    # Performs a single optimization step for parameter updates
    def step(self, lr, theta, grad_val):
        # Calculating gradient('∇f(x,ε)' in paper)
        learn_rate = lr
        dp = grad_val

        # Storing all gradients in a list
        self.gradient.append(dp)

        # Calculating and storing ∑G^2in sqrgradnorm
        self.sqrgradnorm = self.sqrgradnorm + np.power(nlg.norm(dp),2)

        # Updating learning rate('η' in paper)
        power = 1.0/3.0
        scaling = np.power((0.1 + self.sqrgradnorm),power)
        learn_rate = learn_rate/(float)(scaling)

        # Calculating 'a' mentioned as a=cη^2 in paper(denoted 'c' as factor here)
        a = min(self.c*learn_rate**2.0,1.0)

        # Calculating and storing the momentum term(d'=∇f(x',ε')+(1-a')(d-∇f(x,ε')))
        
        if self.count == 0:
            self.momentum = dp
        else:
            self.momentum = self.gradient[-1] + (1-a)*(self.momentum-self.gradient[-2])

        # Updation of theta                
        new_theta = theta - learn_rate*self.momentum
        self.count = self.count + 1
        
        return new_theta