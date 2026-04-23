import numpy as np
import vbgmm


def perform_pca(d, nc, seed):
    # Rust handles centering, SVD (LAPACK dgesdd), sign correction,
    # and component selection (src/pca.rs).
    d.columns = d.columns.astype(str)
    data = np.ascontiguousarray(d.values, dtype=np.float64)
    transformed, components, n_components = vbgmm.perform_pca(data, float(nc))
    return transformed, _PcaComponents(components)


class _PcaComponents:
    """Minimal wrapper to expose components_ for output writing."""
    def __init__(self, components):
        self.components_ = components
