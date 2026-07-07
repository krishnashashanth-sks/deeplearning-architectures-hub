import numpy as np

# --- Karnik-Mendel (KM) Algorithm ---
def km_algorithm(x_values, umf_values, lmf_values, epsilon=1e-6, max_iterations=100):
    num_points = len(x_values)
    if num_points == 0:
        return 0.0, 0.0

    sorted_indices = np.argsort(x_values)
    x_sorted = x_values[sorted_indices]
    umf_sorted = umf_values[sorted_indices]
    lmf_sorted = lmf_values[sorted_indices]

    def calculate_cL():
        k_prev = -1
        k_curr = 0
        
        # Initial centroid calculation using LMF for all points
        numerator_initial = np.sum(x_sorted * lmf_sorted)
        denominator_initial = np.sum(lmf_sorted)
        if denominator_initial == 0:
            return 0.0
        initial_centroid = numerator_initial / denominator_initial

        # Find initial switching point k_curr
        for i in range(num_points):
            if x_sorted[i] >= initial_centroid:
                k_curr = i
                break
        else: # if loop completes without break, initial_centroid is beyond all x_sorted
            k_curr = num_points - 1
        if k_curr == 0 and x_sorted[k_curr] > initial_centroid: # if initial_centroid is before all x_sorted
            k_curr = 0

        for _ in range(max_iterations):
            k_prev = k_curr
            current_mfs = np.zeros(num_points)
            current_mfs[:k_prev+1] = umf_sorted[:k_prev+1]
            current_mfs[k_prev+1:] = lmf_sorted[k_prev+1:]

            numerator = np.sum(x_sorted * current_mfs)
            denominator = np.sum(current_mfs)
            if denominator == 0:
                return 0.0
            centroid_val = numerator / denominator

            k_new = k_prev
            for i in range(num_points):
                if x_sorted[i] >= centroid_val:
                    k_new = i
                    break
            else:
                k_new = num_points - 1
            if k_new == 0 and x_sorted[k_new] > centroid_val:
                k_new = 0

            if abs(k_new - k_prev) < epsilon or _ == max_iterations - 1:
                return centroid_val
            k_curr = k_new
        return centroid_val # Return current best if max_iterations reached

    def calculate_cR():
        k_prev = -1
        k_curr = num_points - 1
        
        # Initial centroid calculation using UMF for all points
        numerator_initial = np.sum(x_sorted * umf_sorted)
        denominator_initial = np.sum(umf_sorted)
        if denominator_initial == 0:
            return 0.0
        initial_centroid = numerator_initial / denominator_initial

        # Find initial switching point k_curr
        for i in range(num_points - 1, -1, -1):
            if x_sorted[i] <= initial_centroid:
                k_curr = i
                break
        else:
            k_curr = 0
        if k_curr == num_points - 1 and x_sorted[k_curr] < initial_centroid:
            k_curr = num_points - 1

        for _ in range(max_iterations):
            k_prev = k_curr
            current_mfs = np.zeros(num_points)
            current_mfs[:k_prev+1] = lmf_sorted[:k_prev+1]
            current_mfs[k_prev+1:] = umf_sorted[k_prev+1:]

            numerator = np.sum(x_sorted * current_mfs)
            denominator = np.sum(current_mfs)
            if denominator == 0:
                return 0.0
            centroid_val = numerator / denominator

            k_new = k_prev
            for i in range(num_points - 1, -1, -1):
                if x_sorted[i] <= centroid_val:
                    k_new = i
                    break
            else:
                k_new = 0
            if k_new == num_points - 1 and x_sorted[k_new] < centroid_val:
                k_new = num_points - 1

            if abs(k_new - k_prev) < epsilon or _ == max_iterations - 1:
                return centroid_val
            k_curr = k_new
        return centroid_val # Return current best if max_iterations reached

    cL = calculate_cL()
    cR = calculate_cR()
    return cL, cR