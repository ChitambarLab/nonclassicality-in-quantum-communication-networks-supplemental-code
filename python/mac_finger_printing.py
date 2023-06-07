import multiple_access_channels as mac
import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo as qnet

def _gradient_descent_wrapper(*opt_args, **opt_kwargs):
    """Wraps ``qnetvo.gradient_descent`` in a try-except block to gracefully
    handle errors during computation.
    This function is called with the same parameters as ``qnetvo.gradient_descent``.
    Optimization errors will result in an empty optimization dictionary.
    """
    try:
        opt_dict = qnet.gradient_descent(*opt_args, **opt_kwargs)
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

def optimize_inequality(nodes, postmap, inequality, **gradient_kwargs):

    mac_ansatz = qnet.NetworkAnsatz(*nodes)

    def opt_fn(placeholder_param):

        print("\nclassical bound : ", inequality[0])

        settings = mac_ansatz.rand_network_settings()
        cost = qnet.linear_probs_cost_fn(mac_ansatz, inequality[1], postmap)
        opt_dict = _gradient_descent_wrapper(cost, settings, **gradient_kwargs)

        print("\nmax_score : ", max(opt_dict["scores"]))
        print("violation : ", max(opt_dict["scores"]) - inequality[0])

        return opt_dict


    return opt_fn

def bipartite_fp_inequality(X):
    fp_game = np.zeros((2,X**2))
    col_id = 0
    for x1 in range(X):
        for x2 in range(X):
            if x1 == x2:
                fp_game[0, col_id] = X-2
            else:
                fp_game[1, col_id] = 1

            col_id += 1
    
    bound = X*(X-2) + (X-1)*2
    return bound, fp_game

def npartite_fp_inequality(n, X=3):
    fp_game = np.zeros((2, X**n))

    # match_ids = [x*X**(n-1) + x**(n-1) + x for x in range(X)]
    match_ids = [sum([x*X**(n-i) for i in range(1,n+1)]) for x in range(X)]

    for col_id in range(X**n):
        if col_id in match_ids:
            fp_game[0, col_id] = 1
        else:
            fp_game[1,col_id] = 1

        col_id += 1

    bound = 0

    return bound, fp_game

def parity_postmap(n):
    postmap = np.zeros((2, 2**n))
    for i in range(2**n):
        row_id = sum([int(digit) for digit in np.binary_repr(i, width=n)]) % 2
        postmap[row_id, i]
    
    return postmap


if __name__=="__main__":


    data_dir = "data/mac_finger-printing/"


    # # parity_postmap = np.array([[1,0,0,1],[0,1,1,0]])
    # and_postmap = np.array([[1,1,1,0],[0,0,0,1]])
    # parity_postmap3 = np.array([
    #     [1,0,0,1,0,1,1,0],
    #     [0,1,1,0,1,0,0,1],
    # ])
    # and_postmap3 = np.array([
    #     [1,1,1,1,1,1,1,0],
    #     [0,0,0,0,0,0,0,1],
    # ])

    def bipartite_qmac_arb_nodes(X):
        qmac_prep_nodes = [
            qnet.PrepareNode(num_in=X, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
            qnet.PrepareNode(num_in=X, wires=[1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2)
        ]
        qmac_meas_nodes = [
            qnet.MeasureNode(num_in=1, num_out=2, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15)
        ]

        return qmac_prep_nodes, qmac_meas_nodes
    
    def npartite_qmac_arb_nodes(n, X):
        qmac_prep_nodes = [
            qnet.PrepareNode(num_in=X, wires=[i], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2)
            for i in range(n)            
        ]
        qmac_meas_nodes = [
            qnet.MeasureNode(num_in=1, num_out=2, wires=range(n), ansatz_fn=qml.ArbitraryUnitary, num_settings=4**n-1)
        ]

        return qmac_prep_nodes, qmac_meas_nodes

    def bipartite_eacmac_arb_nodes(X):
        prep_nodes = [
            qnet.PrepareNode(wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        ]
        ea_meas_nodes = [
            qnet.MeasureNode(num_in=X, num_out=2, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
            qnet.MeasureNode(num_in=X, num_out=2, wires=[1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        ]

        return prep_nodes, ea_meas_nodes
    
    def npartite_eacmac_arb_nodes(n,X):
        prep_nodes = [
            qnet.PrepareNode(wire=range(n), ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2**(n+1)-2),
        ]
        ea_meas_nodes = [
            qnet.MeasureNode(num_in=X, num_out=2, wires=[i], ansatz_fn=qml.ArbitraryUnitary, num_settings=3)
            for i in range(n)
        ]

        return prep_nodes, ea_meas_nodes
    

    
    for X in range(3,4):
        n=3
        inequality = npartite_fp_inequality(n, X)
        print(inequality[1])

        print("X = ", X)
        inequality_tag = "I_fp_X_" + str(X) + "_n_" + str(n)


        postmap = parity_postmap(n)

        n_workers = 1
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

        """
        Bipartite arb QMAC
        """
        client.restart()

        time_start = time.time()

        qmac_arb_opt_fn = optimize_inequality(
            npartite_qmac_arb_nodes(n, X),
            postmap,
            inequality,
            num_steps=150,
            step_size=0.2,
            sample_width=1,
            verbose=False
        )

        qmac_arb_opt_jobs = client.map(qmac_arb_opt_fn, range(n_workers))
        qmac_arb_opt_dicts = client.gather(qmac_arb_opt_jobs)

        max_opt_dict = qmac_arb_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qmac_arb_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qmac_arb_opt_dicts[j]["scores"])
                max_opt_dict = qmac_arb_opt_dicts[j]

        scenario = "bipartite_qmac_arb_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnet.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        Bipartite  EACMAC
        """
        client.restart()

        time_start = time.time()

        eacmac_arb_opt_fn = optimize_inequality(
            npartite_eacmac_arb_nodes(n,X),
            postmap,
            inequality,
            num_steps=150,
            step_size=0.2,
            sample_width=1,
            verbose=False
        )

        eacmac_arb_opt_jobs = client.map(eacmac_arb_opt_fn, range(n_workers))
        eacmac_arb_opt_dicts = client.gather(eacmac_arb_opt_jobs)

        max_opt_dict = eacmac_arb_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eacmac_arb_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eacmac_arb_opt_dicts[j]["scores"])
                max_opt_dict = eacmac_arb_opt_dicts[j]

        scenario = "bipartite_eacmac_arb_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnet.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_ta + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)