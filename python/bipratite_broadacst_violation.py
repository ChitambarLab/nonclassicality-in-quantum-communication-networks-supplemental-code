import multiple_access_channels as mac
import pennylane as qml
from pennylane import numpy as np
# from dask.distributed import Client
import time
from datetime import datetime
import json
import copy

import qnetvo

def _gradient_descent_wrapper(*opt_args, **opt_kwargs):
    """Wraps ``qnetvo.gradient_descent`` in a try-except block to gracefully
    handle errors during computation.
    This function is called with the same parameters as ``qnetvo.gradient_descent``.
    Optimization errors will result in an empty optimization dictionary.
    """
    try:
        opt_dict = qnetvo.gradient_descent(*opt_args, **opt_kwargs)
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

def optimize_inequality(prep_nodes, meas_nodes, postmap, inequality, **gradient_kwargs):

    broadcast_ansatz = qnetvo.NetworkAnsatz(prep_nodes, meas_nodes)

    def opt_fn(placeholder_param):

        print("\nclassical bound : ", inequality[0])

        settings = broadcast_ansatz.rand_network_settings()
        cost = qnetvo.linear_probs_cost_fn(broadcast_ansatz, inequality[1], postmap)
        opt_dict = _gradient_descent_wrapper(cost, settings, **gradient_kwargs)

        print("\nmax_score : ", opt_dict["opt_score"])
        print("violation : ", opt_dict["opt_score"] - inequality[0])

        return opt_dict


    return opt_fn


def write_optimization_json(opt_dict, filename):
    """Writes the optimization dictionary to a JSON file.

    :param opt_dict: The dictionary returned by a network optimization.
    :type opt_dict: dict

    :param filename: The name of the JSON file to be written. Note that ``.json`` extension is automatically added.
    :type filename: string

    :returns: ``None``
    """

    opt_dict_json = copy.deepcopy(opt_dict)

    opt_dict_json["opt_score"] = float(opt_dict_json["opt_score"])
    opt_dict_json["scores"] = [float(score) for score in opt_dict_json["scores"]]
    opt_dict_json["opt_settings"] = [setting.tolist() for setting in opt_dict_json["opt_settings"]]
    opt_dict_json["settings_history"] = [
        [setting.tolist() for setting in settings] for settings in opt_dict_json["settings_history"]
    ]

    with open(filename + ".json", "w") as file:
        file.write(json.dumps(opt_dict_json, indent=2))


if __name__=="__main__":


    data_dir = "data/bipartite_broadcast_quantum_violations/"

    def prep_ansatz(settings, wires):
        # qml.ArbitraryUnitary(settings[0:3], [wires[0]])
        # qml.ArbitraryUnitary(settings[3:6], [wires[3]])

        qml.ArbitraryUnitary(settings[0:15], [wires[0], wires[3]])
        # qnetvo.ghz_state([], wires=[wires[1],wires[2]])

    prep_nodes = [
        qnetvo.PrepareNode(num_in=4, wires=[0,1,2,3], ansatz_fn=prep_ansatz, num_settings=15)
    ]
    meas_nodes = [
        qnetvo.MeasureNode(num_out=4, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=4, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15)
    ]

    # true entanglement-assistance witness
    #
    # 8.5 violation entanglement-assisted separable broadcast
    # 8.5 violataion when entanglement assists general broadcast
    inequality = (8, np.array(
        [
            [3,0,0,0],
            [0,3,0,0],
            [0,0,0,3],
            [1,1,0,2],
            [1,1,0,2],
            [1,1,0,2],
            [2,1,0,2],
            [1,3,0,2],
            [1,0,2,0],
            [0,1,2,0],
            [0,0,0,3],
            [1,1,0,2],
            [2,1,0,2],
            [1,2,0,2],
            [1,1,1,2],
            [2,2,1,2],
        ]
    ))

    # # correlated signal communication value
    # # not violated in any qubit broadcast scenario
    # inequality = (2, np.array([
    #     [1,0,0,0],
    #     [0,1,0,0],
    #     [0,0,0,0],
    #     [0,0,0,0],
    #     [0,1,0,0],
    #     [1,0,0,0],
    #     [0,0,0,0],
    #     [0,0,0,0],
    #     [0,0,0,0],
    #     [0,0,0,0],
    #     [0,0,1,0],
    #     [0,0,0,1],
    #     [0,0,0,0],
    #     [0,0,0,0],
    #     [0,0,0,1],
    #     [0,0,1,0],
    # ]))

    # # embedded chsh game
    # # 2 : not violated by separable quantum broadcast
    # # 3 : violated by entangled broadcast
    # # 2 sqrt(2) : violataed by entanglement assisted separable brodacast
    # # 3 : violated by entangelement assisted quantum broadcast 
    # inequality = (2, np.array([
    #     [1,0,0,0],
    #     [0,1,0,0],
    #     [-1,0,0,0],
    #     [0,-1,0,0],
    #     [-1,0,0,0],
    #     [0,-1,0,0],
    #     [1,0,0,0],
    #     [0,1,0,0],
    #     [0,0,1,0],
    #     [0,0,0,-1],
    #     [0,0,-1,0],
    #     [0,0,0,1],
    #     [0,0,-1,0],
    #     [0,0,0,1],
    #     [0,0,1,0],
    #     [0,0,0,-1],
    # ]))

    postmap = np.eye(16)

    opt_dict = optimize_inequality(prep_nodes, meas_nodes, postmap, inequality,
        sample_width=1,
        step_size=0.2,
        num_steps=70,
        verbose=True,
    )([])