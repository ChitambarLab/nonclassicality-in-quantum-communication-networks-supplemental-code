import multiple_access_channels as mac
import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

from context import qnetvo as qnet

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

def optimize_inequality(prep_nodes, meas_nodes, postmap, inequality, **gradient_kwargs):

    mac_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

    def opt_fn(placeholder_param):

        print("\nclassical bound : ", inequality[0])

        settings = mac_ansatz.rand_scenario_settings()
        cost = qnet.linear_probs_cost_fn(mac_ansatz, inequality[1], postmap)
        opt_dict = _gradient_descent_wrapper(cost, settings, **gradient_kwargs)

        print("\nmax_score : ", opt_dict["opt_score"])
        print("violation : ", opt_dict["opt_score"] - inequality[0])

        return opt_dict


    return opt_fn


if __name__=="__main__":


    data_dir = "data/bisender_mac_quantum_violations/"


    parity_postmap = np.array([[1,0,0,1],[0,1,1,0]])
    and_postmap = np.array([[1,1,1,0],[0,0,0,1]])

    qmac_prep_nodes = [
        qnet.PrepareNode(3, [0], qml.ArbitraryStatePreparation, 2),
        qnet.PrepareNode(3, [1], qml.ArbitraryStatePreparation, 2)
    ]
    qmac_meas_nodes = [
        qnet.MeasureNode(1, 2, [0,1], qml.ArbitraryUnitary, 15)
    ]

    ea_mac_prep_nodes = [
        qnet.PrepareNode(1, [0,1], qml.ArbitraryStatePreparation, 6)
    ]
    ea_mac_meas_nodes = [
        qnet.MeasureNode(
            3, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        ),
        qnet.MeasureNode(
            3, 2, [1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        )
    ]

    inequalities = [(7, mac.finger_printing_matrix(2,3))] + mac.bisender_mac_bounds()
    
    for i in range(12,20):
        inequality = inequalities[i]

        print("i = ", i)
        inequality_tag = "I_fp_" if i == 0 else "I_" + str(i) + "_"


        for postmap_tag in ["xor_", "and_"]:
            postmap = parity_postmap if postmap_tag == "xor_" else and_postmap

            client = Client(processes=True, n_workers=5, threads_per_worker=1)

            """
            QMAC
            """
            time_start = time.time()

            qmac_opt_fn = optimize_inequality(
                qmac_prep_nodes,
                qmac_meas_nodes,
                postmap,
                inequality,
                num_steps=150,
                step_size=0.15,
                sample_width=1,
                verbose=False
            )

            qmac_opt_jobs = client.map(qmac_opt_fn, range(5))
            qmac_opt_dicts = client.gather(qmac_opt_jobs)

            max_opt_dict = qmac_opt_dicts[0]
            max_score = max(max_opt_dict["scores"])
            for j in range(1,5):
                if max(qmac_opt_dicts[j]["scores"]) > max_score:
                    max_score = max(qmac_opt_dicts[j]["scores"])
                    max_opt_dict = qmac_opt_dicts[j]

            scenario = "qmac_"
            datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            qnet.write_optimization_json(
                max_opt_dict,
                data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            )

            print("iteration time  : ", time.time() - time_start)

            """
            EA MAC
            """
            time_start = time.time()

            ea_mac_opt_fn = optimize_inequality(
                ea_mac_prep_nodes,
                ea_mac_meas_nodes,
                postmap,
                inequality,
                num_steps=150,
                step_size=0.15,
                sample_width=1,
                verbose=False
            )

            ea_mac_opt_jobs = client.map(ea_mac_opt_fn, range(5))
            ea_mac_opt_dicts = client.gather(ea_mac_opt_jobs)

            max_opt_dict = ea_mac_opt_dicts[0]
            max_score = max(max_opt_dict["scores"])
            for j in range(1,5):
                if max(ea_mac_opt_dicts[j]["scores"]) > max_score:
                    max_score = max(ea_mac_opt_dicts[j]["scores"])
                    max_opt_dict = ea_mac_opt_dicts[j]

            scenario = "ea_mac_"
            datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            qnet.write_optimization_json(
                max_opt_dict,
                data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            )

            print("iteration time  : ", time.time() - time_start)
