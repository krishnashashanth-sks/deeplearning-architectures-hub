import numpy as np

# ---  Type-2 Trapezoidal Membership Function (Conceptual for Fuzzification) ---
def trapezoidal_mf(x, a, b, c, d):
    # Type-1 trapezoidal MF function
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)))

def create_type2_trapezoidal_mf_params(center, width, uncertainty_factor):
    # Simulate uncertainty around a Type-1 trapezoidal MF.
    # The 'uncertainty_factor' determines the spread of the FOU.

    # Nominal Type-1 MF parameters
    a_nom, b_nom, c_nom, d_nom = center - width/2, center - width/4, center + width/4, center + width/2

    # UMF parameters (wider)
    a_umf = a_nom - uncertainty_factor * width/4
    b_umf = b_nom - uncertainty_factor * width/8
    c_umf = c_nom + uncertainty_factor * width/8
    d_umf = d_nom + uncertainty_factor * width/4

    # LMF parameters (narrower, typically within UMF)
    a_lmf = a_nom + uncertainty_factor * width/8
    b_lmf = b_nom + uncertainty_factor * width/16
    c_lmf = c_nom - uncertainty_factor * width/16
    d_lmf = d_nom - uncertainty_factor * width/8

    # Ensure b_lmf > a_lmf and d_lmf > c_lmf
    b_lmf = max(a_lmf + 1e-6, b_lmf)
    d_lmf = max(c_lmf + 1e-6, d_lmf)

    return (a_umf, b_umf, c_umf, d_umf), (a_lmf, b_lmf, c_lmf, d_lmf)
