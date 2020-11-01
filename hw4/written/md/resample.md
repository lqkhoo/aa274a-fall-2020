```py
def resample(self, xs, ws):
    """
    Resamples the particles according to the updated particle weights.
    Inputs:
        xs: np.array[M,3] - matrix of particle states.
        ws: np.array[M,]  - particle weights.
    Output:
        None - internal belief state (self.xs, self.ws) should be updated.
    """
    r = np.random.rand() / self.M
    ########## Code starts here ##########
    # The way to see the algorithm is that the random value of r generates
    # a sampling 'sieve' which we then use to pick out particles which are
    # represented in terms of their weight on a sampling interval [0, 1].
    # This sieve has as many points as we have particles.

    # r ~ U[0, 1/n]
    n = self.M
    m = np.linspace(0, n, n, endpoint=False) # {0, ..., n-1}
    sieve = r + m/n
    u = np.sum(ws) * sieve # Normalization step. Maintains [0, 1]
    csum = np.cumsum(ws)
    idx = np.searchsorted(csum, u)
    self.xs = xs[idx]
    self.ws = ws[idx]
    
    ########## Code ends here ##########
```
