import numpy as np

# pysynphot uses np.alltrue which was removed in NumPy 2.0. Restore the alias
# so the package can be imported while the broader incompatibility is unresolved.
if not hasattr(np, "alltrue"):
    np.alltrue = np.all
