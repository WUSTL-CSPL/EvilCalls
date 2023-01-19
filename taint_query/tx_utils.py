import numpy as np
import scipy.io.wavfile as wav
from numpy import linalg as LA


def save_wav(audio, output_wav_file):
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16))

def cal_mean_sqr(theta):
    lbd = np.sqrt(np.mean(np.absolute(theta)**2))
    return lbd

class SignOpt():
    def __init__(self, x0, target, play_name, output_path, use_storm = False):
        self.K = 10         
        self.D = 4
        self.beta = 0.001
        self.alpha = 0.2
        self.x0 = x0
        self.target = target
        self.play_name = play_name
        self.output_file_path = output_path

        #intermediate variables for global binary search
        self.perb_audio = np.zeros(x0.shape, dtype=np.float64)
        self.global_lbd_hi = 0
        self.global_lbd_lo = 0
        self.ini_flag = 0
        self.alpha_temp = 0
        self.initial_theta = np.zeros(x0.shape, dtype=np.float64)
        
        #intermediate variables for sign opt search
        self.initial_lbd = 0
        self.xg = np.zeros(x0.shape, dtype=np.float64)
        self.gg = 0
        self.sign_opt_stage = 0
        self.perb_count = 0

        #intermediate variables for sign grad
        self.sign_gradient = np.zeros(x0.shape, dtype=np.float64)
        self.grad_count = -1

        #intermediate variables for local binary search
        self.local_lbd_lo = 0
        self.local_lbd_hi = 0
        self.stage_flag = 0
        self.local_lo_flag = 0

    def global_binary_search(self, decision):
        """
        perform the global binary search first
        """
        #calculate theta if enter the function for the first time
        if self.ini_flag == 0:
            self.initial_theta = self.perb_audio - self.x0   
            initial_lbd = cal_mean_sqr(self.initial_theta)
            self.initial_theta /= initial_lbd
            self.global_lbd_lo = 0
            self.global_lbd_hi = 2 * initial_lbd
            decision = self.target
            self.ini_flag = 1
        
        decision = decision.upper()
        lbd_mid = (self.global_lbd_hi + self.global_lbd_lo)/2.0

        if decision != self.target:
            self.global_lbd_lo = lbd_mid
        else:
            self.global_lbd_hi = lbd_mid

        lbd_mid = (self.global_lbd_hi + self.global_lbd_lo)/2.0
        save_wav(self.x0 + lbd_mid*self.initial_theta, self.play_name)
        return

    def sign_opt_search(self, decision):
        """
        main sign-opt logic, use gradient to update theta and perform local binary search
        """
        if self.sign_opt_stage == 0:
            if self.grad_count < self.K-1:
                self.sign_grad(decision, self.initial_lbd, self.initial_theta)
            else: 
                self.sign_gradient = self.sign_grad(decision, self.initial_lbd, self.initial_theta)
                self.sign_opt_stage = 1
                self.alpha_temp = self.alpha

        if self.sign_opt_stage == 1:
            min_theta = self.xg
            min_g2 = self.gg

            if self.perb_count < 10:
                
                #update theta
                new_theta = self.xg - self.alpha_temp * self.sign_gradient
                new_theta /= cal_mean_sqr(new_theta)

                new_g2 = self.local_binary_search(decision, min_g2, new_theta, self.beta/500)
                if new_g2 == 0:
                    return 0
                else:
                    #take small steps to search
                    self.alpha_temp = self.alpha_temp * 0.5
                    self.perb_count += 1
                    if new_g2 < self.gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        self.sign_opt_stage = 2
                        
            if self.sign_opt_stage == 2 or self.perb_count >= 10:
                if self.alpha_temp < 1e-5:
                    print("Warning: not moving.")
                    self.beta = self.beta * 0.1
                    if self.beta < 1e-8:
                        save_wav(self.x0 + min_g2*min_theta, self.output_file_path)
                        return 1
                self.xg = min_theta
                self.gg = min_g2

                save_wav(self.x0 + min_g2*min_theta, "temp.wav")

                self.sign_opt_stage = 0
                self.grad_count = -1
                self.perb_count = 0

        return 0


    def sign_grad(self, decision, initial_lbd, theta):
        """
        calculate sign-opt gradient
        """
        u = np.random.randn(*theta.shape)
        u /= LA.norm(u)
        new_theta = theta + self.beta*u
        temp_norm = LA.norm(new_theta)
        new_theta /= temp_norm
        
        save_wav(self.x0+initial_lbd*new_theta*temp_norm, self.play_name)
        
        if self.grad_count > -1:
            sign = 1
            if decision == self.target:
                sign = -1
            self.sign_gradient += np.sign(u)*sign

        self.grad_count += 1

        if self.grad_count == self.K:
            grad_val = self.sign_gradient/self.K
            self.sign_gradient = np.zeros(theta.shape)
            self.grad_count = -1
            return grad_val
        
        return np.zeros(theta.shape)


    def local_binary_search(self, decision, initial_lbd, theta, tol=1e-5):
        """
        perform local binary search
        """
        # when first comes, just play wave 
        if self.stage_flag == 0:
            save_wav(self.x0 + initial_lbd*theta, self.play_name)
            self.stage_flag = 1
            return 0

        decision = decision.upper()
        if self.stage_flag == 1:
            #stage 1: determine lbd_lo and lbd_hi
            if decision != self.target and self.local_lo_flag == 0:
                return float('inf')
            else:
                if self.local_lo_flag == 0:
                    self.local_lbd_hi = initial_lbd
                    self.local_lbd_lo = initial_lbd*0.99
                    self.local_lo_flag = 1
                else:
                    if decision == self.target:
                        self.local_lbd_lo = self.local_lbd_lo*0.99
                    else:
                        self.local_lo_flag = 0
                        self.stage_flag = 2
        else:
            #stage 2: find the minimum lbd
            if (self.local_lbd_hi - self.local_lbd_lo) <= tol:
                self.stage_flag = 0
                return self.local_lbd_hi
            
            lbd_mid = (self.local_lbd_hi + self.local_lbd_lo)/2.0
            if decision != self.target:
                self.local_lbd_lo = lbd_mid
            else:
                self.local_lbd_hi = lbd_mid

            lbd_mid = (self.local_lbd_hi + self.local_lbd_lo)/2.0
            save_wav(self.x0 + lbd_mid*theta, self.play_name)
        
        return 0

         





