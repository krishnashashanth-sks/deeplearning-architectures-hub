import matplotlib.pyplot as plt

# --- Visualization for Type-2 Trapezoidal Membership Function ---
def plot_type2_mf(x_values, umf_memberships, lmf_memberships, feature_value=None, title='Conceptual Type-2 Trapezoidal Membership Function'):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, umf_memberships, label='UMF')
    plt.plot(x_values, lmf_memberships, label='LMF')
    plt.fill_between(x_values, lmf_memberships, umf_memberships, color='gray', alpha=0.3, label='FOU')
    if feature_value is not None:
        plt.axvline(x=feature_value, color='red', linestyle='--', label=f'Crisp Feature: {feature_value:.2f}')
    plt.title(title)
    plt.xlabel('Feature Value')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()

