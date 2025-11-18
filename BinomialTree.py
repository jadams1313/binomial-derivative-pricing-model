import numpy as np 

class BinomialTree:
    def __init__(self, S_0, T, K, r, u , d, N, usesCoxRosenstein, sigma=0):
        self.S_0 = S_0  # Initial stock price
        self.T = T      # Time to maturity
        self.K = K      # Strike price
        self.r = r      # Risk-free interest rate
        self.sigma = sigma  # Volatility
        self.N = N      # Number of time steps
        self.dt = T / N  # Time step size
        self.u = np.exp(sigma * np.sqrt(self.dt))  # Up factor
        self.d = 1 / self.u  # Down factor
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral probability

    def stock_price(self, i, j):
        """Calculate the stock price at node (i, j)"""
        return round(self.S_0 * (self.u ** j) * (self.d ** (i - j)), 4)
     
    def payoff(self, S, derivative_type):
        """Calculate the payoff of the derivative at maturity"""
        if derivative_type == 'call':
            return max(0, S - self.K)
        else:
            return max(0, self.K - S)
        
    def option_price(self, derivative_type):
        """Calculate the option price using the binomial tree model"""
        payoffs = np.zeros(self.N + 1)
        values = np.zeros(self.N + 1)

        # initialize asset prices at maturity
        payoffs = np.zeros(self.N + 1)
    
        # Initialize asset prices at maturity
        for i in range(self.N + 1):
            S = self.stock_price(self.N, i)
            payoffs[i] = self.payoff(S, derivative_type=derivative_type)
        
        # Backward induction through the tree
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                payoffs[j] = round(np.exp(-self.r * self.dt) * (
                    self.p * payoffs[j + 1] + (1 - self.p) * payoffs[j]
                ), 4) 
        
        return payoffs[0]

def black_scholes_call_price(S, K, T, r, sigma):
    from scipy.stats import norm
    from math import log, sqrt, exp

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    call_price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return call_price
def put_price(C, S, K, T, r):
    from math import exp
    P = C + K * exp(-r * T) - S
    return P
        

