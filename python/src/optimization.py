import numpy as np
import qnetvo

def _gradient_descent_wrapper(*opt_args, **opt_kwargs):
    """Wraps ``qnetvo.gradient_descent`` in a try-except block to gracefully
    handle errors during computation.
    This function is called with the same parameters as ``qnetvo.gradient_descent``.
    If an optimization error occurs an empty optimization dictionary is returned.
    """
    try:
        opt_dict = qnetvo.gradient_descent(*opt_args, **opt_kwargs, optimizer="adam")
    except Exception as err:
        print("An error occurred during gradient descent.")
        print(err)
        opt_dict = {
            "opt_score": np.nan,
            "opt_settings": [[], []],
            "scores": [np.nan],
            "samples": [0],
            "settings_history": [[[], []]],
        }

    return opt_dict

def optimize_inequality(nodes, postmap, inequality, fixed_setting_ids=[], fixed_settings=[], **gradient_kwargs):
    """
    Returns an `opt_fn()` function that optimizess the specified network ansatz to maximize the score of a given
    linear inequality.
     
    :param nodes: A list of network layer where each layer is a list of `qnetvo` network nodes (`List[qnetvo.Node]`).
    :type nodes: List[List[qnetvo.Node]]

    :param postmap: A column stochastic matrix that maps the ansatz circuit's measured qubit probabilities to the number
                    of outputs of the network device. E.g. a 3x4 matrix maps a two-qubit measurement result to a trit. 
    :type np.ndarray:

    :param inequality: A linear inequality tuple `(gamma, G)` where `gamma` is the upper bound and `G` is the game matrix.
    ;type inequality: Tuple[float, nd.array]

    :param fixed_setting_ids: The indices of all static ansatz settings that are held constant during optimization.
    :type fixed_setting_ids: List[int]

    :param fixed_settins: The value of the static ansatz settings that are held constant during optimization.
    :type fixed_settings: List[float]

    :param gradient_kwargs: The keyworkd arguments passed to the `qnetvo.gradient_descent` function.
    :type gradient_kwargs: Keywork Arguments
    """
    network_ansatz = qnetvo.NetworkAnsatz(*nodes)

    def opt_fn(placeholder_param):

        print("\nclassical bound : ", inequality[0])

        settings = network_ansatz.rand_network_settings(fixed_setting_ids=fixed_setting_ids,fixed_settings=fixed_settings)
        cost = qnetvo.linear_probs_cost_fn(network_ansatz, inequality[1], postmap)
        opt_dict = _gradient_descent_wrapper(cost, settings, **gradient_kwargs)

        print("\nmax_score : ", max(opt_dict["scores"]))
        print("violation : ", max(opt_dict["scores"]) - inequality[0])

        return opt_dict


    return opt_fn