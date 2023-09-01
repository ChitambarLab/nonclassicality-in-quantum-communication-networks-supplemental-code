import multiple_access_channels as mac
import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

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

def optimize_inequality(nodes, postmap, inequality, fixed_setting_ids=[], fixed_settings=[], **gradient_kwargs):

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

def rac_game(n):
    X = 2**n
    Y = n

    game = np.zeros((2, X*Y))
    col_id = 0
    for x in range(X):
        for y in range(Y):
            bin_string = [int(i) for i in np.binary_repr(x, width=n)]

            row_id = bin_string[y]
            game[row_id, col_id] = 1

            col_id += 1
    
    if n == 2:
        bound = 6
    elif n == 3:
        bound = 18
    elif n == 4:
        bound = 28
    else:
        bound = X*Y - 2

    return (bound, game)




if __name__=="__main__":


    data_dir = "data/qubit_random_access_coding/"

    postmap = np.eye(2)

    for n in range(6,7):
        qubit_prep_nodes = [
            qnetvo.PrepareNode(num_in = 2**n, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        ]
    
        qubit_meas_nodes = [
            qnetvo.MeasureNode(num_in=n, num_out=2, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        ]

        rac_inequality = rac_game(n)
        print(rac_inequality[1])


        num_in = rac_inequality[1].shape[1]

        print("n = ", n)
        inequality_tag = "I_n_" + str(n) + "_"

        n_workers = 1
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

        """
        qubit interference
        """
        client.restart()

        time_start = time.time()

        rac_n_opt_fn = optimize_inequality(
            [
                qubit_prep_nodes,
                qubit_meas_nodes,
            ],
            postmap,
            rac_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        rac_n_opt_jobs = client.map(rac_n_opt_fn, range(n_workers))
        rac_n_opt_dicts = client.gather(rac_n_opt_jobs)

        max_opt_dict = rac_n_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(rac_n_opt_dicts[j]["scores"]) > max_score:
                max_score = max(rac_n_opt_dicts[j]["scores"])
                max_opt_dict = rac_n_opt_dicts[j]

        scenario = "rac_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

       