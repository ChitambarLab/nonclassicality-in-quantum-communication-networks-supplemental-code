from context import QNetOptimizer as QNopt
from pennylane import math
from pennylane import numpy as np
import matplotlib.pyplot as plt

def bisender_mac_mutual_info(mac_behavior, priors_x, priors_y):
    """Evaluates the rates and mutual information for the given
    conditional probability distribution ``mac_behavior`` and the
    corresponding input and output prior distributions.
    The mutual information for a bisender multiple access channel is
    characterized by three quantities:

    .. math::

        I(X;Z|Y) &= H(XY) + H(YZ) - H(Y) - H(XYZ) \\\\
        I(Y;Z|X) &= H(XY) + H(XZ) - H(X) - H(XYZ) \\\\
        I(XY;Z) &= H(XY) + H(Z) - H(XYZ)

    
    where :math:`H(X)` is the shannon entropy (see ``shannon_entropy``).

    :param mac_behavior: A column stochastic matrix describing the conditional
        probabilities the bi-sender multiple access channel.
    :type mac_behavior: np.array

    :param priors_x: A discrete probability vector describing the input set X.
    :type priors_x: np.array

    :param priors_y: A discrete probability vector describing the input set Y.
    :type priors_y: np.array

    :returns: A tuple with three values ``(I(X;Z|Y), I(Y;Z|X), I(XY;Z))``.
    :rtype: tuple
    """
    num_z = mac_behavior.shape[0]
    num_x = len(priors_x)
    num_y = len(priors_y)

    # joint probability distributions
    p_xy = math.kron(priors_x, priors_y)
    p_xyz = mac_behavior * p_xy
    p_z = math.array([math.sum(row) for row in p_xyz])

    p_yz = math.zeros((num_z, num_y))
    for x in range(num_x):
        p_yz += p_xyz[:, x * num_y : (x + 1) * num_y]

    p_xz = math.zeros((num_z, num_x))
    for y in range(num_y):
        p_xz += p_xyz[:, y : num_x * num_y : num_y]

    # shannon entropies
    H_x = QNopt.shannon_entropy(priors_x)
    H_y = QNopt.shannon_entropy(priors_y)
    H_z = QNopt.shannon_entropy(p_z)
    H_xy = QNopt.shannon_entropy(p_xy)
    H_xz = QNopt.shannon_entropy(p_xz.reshape(num_x * num_z))
    H_yz = QNopt.shannon_entropy(p_yz.reshape(num_y * num_z))
    H_xyz = QNopt.shannon_entropy(p_xyz.reshape(num_x * num_y * num_z))

    # I(X;Z|Y)
    I_x_zy = H_xy + H_yz - H_y - H_xyz

    # I(Y;Z|X)
    I_y_zx = H_xy + H_xz - H_x - H_xyz

    # I(XY;Z)
    I_xy_z = H_xy + H_z - H_xyz

    return I_x_zy, I_y_zx, I_xy_z

def priors_scan_range(num_steps):
    eps = 1e-10
    x1_range = np.arange(0,1+eps,1/num_steps)

    priors = []
    for x1 in x1_range:
        x2_range = np.arange(0,1-x1+eps,1/num_steps)
        for x2 in x2_range:
            priors.append(np.array([x1,x2,1-x1-x2]))
        
    return priors

def plot_rate_region(rate_tuple):
    r1, r2, r_sum = rate_tuple
    
    r1_vals = [0,0,r_sum-r2,r1,r1,0]
    r2_vals = [0,r2,r2,r_sum-r1,0,0]
    
    plt.plot(r1_vals, r2_vals,"b-",label="Quantum")
    plt.plot([0,0,1,0],[0,1,0,0],"r--",label="Classical")
    plt.legend()
    plt.title("Multiple Access Channel Rate Regions")
    plt.xlabel("Rate 1")
    plt.ylabel("Rate 2")
    plt.show()
