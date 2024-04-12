from skopt.space import Space, Real

#lr, lam, threshold, tau_grad, scale_grad
SEARCH_SPACE = Space([
    Real(.0005, .0015, name="lr"),
    Real(0.009, 0.012, name="lam"),
    Real(0.05, 0.15, name='threshold'),
    Real(0.4, 0.6, name='tau_grad'),
    Real(.9,1.1, name="scale_grad")
])

MINIMA: float = 0.0
