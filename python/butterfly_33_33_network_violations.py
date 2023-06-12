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

if __name__=="__main__":


    data_dir = "data/butterfly_33_33_network_violations/"

    # postmap3 = np.array([
    #     [1,0,0,0],[0,1,0,0],[0,0,1,1],
    # ])
    postmap3 = np.array([
        [1,0,0,1],[0,1,0,0],[0,0,1,0],
    ])

    butterfly_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4])
    ]
    butterfly_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        qnetvo.PrepareNode(num_in=3, wires=[2,4], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    butterfly_B_nodes = [
        qnetvo.ProcessingNode(wires=[1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    butterfly_C_nodes = [
        qnetvo.ProcessingNode(wires=[1,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    butterfly_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[3,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    eatx_butterfly_source_nodes = [
        qnetvo.PrepareNode(wires=[1,2], ansatz_fn=qnetvo.ghz_state)
    ]
    eatx_butterfly_prep_nodes = [
        qnetvo.ProcessingNode(num_in=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.ProcessingNode(num_in=3, wires=[2,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]


    butterfly_game_inequalities = [
        (7, np.array([ # multiplication with zero 
            [1, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])),
        (6 , np.array([ # multiplication game [1,2,3] no zero mult1
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])),
        (4, np.array([ # swap game
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])),
        (6, np.array([ # adder game
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])),
        (7, np.array([ # compare game
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])),
        (6, np.array([ # on receiver permutes output based on other receiver
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ])),
        (6, np.array([ # same difference game
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
        ])),
        (7, np.eye(9)), # communication value
    ]
    
    butterfly_facet_inequalities = [
        (11, np.array([ # mult0 facet
            [1,  3,  1,  1,  0,  0,  1,  0,  0],
            [0,  0,  0,  1,  2,  0,  1,  0,  1],
            [0,  1,  0,  0,  0,  1,  0,  2,  0],
            [0,  2,  1,  0,  1,  1,  0,  1,  1],
            [0,  1,  0,  1,  2,  0,  0,  0,  2],
            [0,  2,  1,  1,  1,  1,  0,  1,  1],
            [0,  2,  1,  0,  1,  1,  0,  1,  1],
            [0,  0,  0,  1,  1,  0,  1,  0,  1],
            [1,  2,  1,  1,  1,  1,  1,  1,  1],
        ])),
        (11, np.array([ # mult1 facet
            [2,  0,  1,  1,  0,  0,  0,  0,  0],
            [0,  1,  1,  3,  0,  0,  0,  0,  0],
            [0,  0,  1,  1,  0,  0,  2,  0,  0],
            [1,  0,  1,  0,  1,  1,  0,  1,  0],
            [0,  1,  1,  1,  0,  2,  0,  1,  0],
            [0,  0,  1,  1,  0,  2,  0,  2,  0],
            [1,  1,  1,  1,  1,  1,  0,  0,  1],
            [0,  1,  0,  2,  1,  1,  1,  0,  1],
            [1,  1,  1,  2,  1,  1,  1,  1,  1],
        ])),
        (12, np.array([ # swap facet
            [2,  0,  1,  0,  0,  1,  0,  1,  0],
            [0,  0,  1,  2,  0,  1,  0,  1,  0],
            [1,  0,  1,  1,  1,  1,  1,  1,  1],
            [0,  2,  0,  0,  0,  1,  1,  0,  0],
            [0,  0,  1,  0,  2,  0,  1,  0,  0],
            [0,  1,  1,  1,  1,  1,  0,  1,  1],
            [0,  0,  3,  0,  0,  2,  0,  1,  0],
            [0,  0,  2,  0,  0,  3,  0,  1,  0],
            [1,  1,  2,  1,  1,  2,  1,  1,  1],
        ])),
        (10, np.array([ # adder facet
            [1,  0,  1,  0,  1,  1,  1,  1,  0],
            [0,  2,  0,  1,  0,  0,  0,  0,  2],
            [1,  1,  2,  0,  1,  1,  1,  1,  1],
            [0,  1,  1,  0,  0,  2,  1,  1,  0],
            [0,  1,  0,  0,  0,  0,  0,  0,  2],
            [0,  1,  2,  0,  0,  1,  1,  1,  1],
            [1,  0,  1,  0,  1,  1,  1,  1,  1],
            [0,  1,  0,  1,  0,  0,  0,  0,  2],
            [1,  1,  2,  0,  1,  1,  1,  1,  1],
        ])),
        (14, np.array([ # compare facet
            [3,  0,  0,  0,  2,  0,  1,  0, 1],
            [2,  1,  0,  0,  1,  0,  1,  1, 1],
            [2,  1,  1,  0,  0,  1,  0,  1, 0],
            [1,  1,  2,  0,  0,  2,  0,  1, 1],
            [0,  2,  2,  0,  0,  2,  0,  1, 1],
            [0,  3,  2,  0,  0,  3,  0,  0, 0],
            [1,  1,  2,  0,  1,  1,  1,  1, 0],
            [1,  2,  2,  1,  1,  1,  1,  1, 0],
            [2,  2,  3,  1,  1,  2,  1,  1, 0],
        ])),
        (20, np.array([ # conditioned permutation facet
            [2,  1,  0,  1,  3,  0,  1,  1,  2],
            [1,  5,  2,  0,  0,  2,  1,  1,  0],
            [0,  3,  3,  1,  3,  3,  0,  2,  0],
            [1,  2,  0,  1,  4,  0,  1,  0,  2],
            [0,  3,  1,  1,  1,  3,  1,  1,  0],
            [0,  3,  3,  2,  1,  2,  0,  1,  0],
            [1,  0,  0,  1,  3,  0,  1,  1,  1],
            [0,  3,  1,  0,  0,  2,  2,  1,  1],
            [2,  3,  3,  1,  3,  3,  1,  2,  0],
        ])),
        (12, np.array([  # same difference facet
            [3,  0,  0,  0,  2,  0,  0,  1,  1],
            [1,  0,  0,  1,  1,  0,  0,  1,  1],
            [2,  0,  0,  0,  1,  0,  0,  0,  1],
            [2,  0,  1,  0,  0,  2,  0,  1,  0],
            [1,  0,  1,  2,  0,  1,  0,  2,  0],
            [2,  1,  1,  1,  1,  2,  0,  1,  0],
            [1,  0,  0,  0,  0,  1,  0,  1,  1],
            [0,  0,  1,  1,  0,  2,  0,  1,  1],
            [2,  1,  2,  1,  1,  2,  1,  1,  0],
        ])),
        (18, np.array([ # cv facet
            [3,  0,  1,  0,  3,  1,  0,  0,  0],
            [1,  3,  0,  0,  3,  1,  0,  0,  0],
            [0,  0,  3,  0,  4,  1,  0,  0,  1],
            [1,  0,  1,  3,  2,  1,  0,  0,  0],
            [3,  1,  1,  1,  3,  0,  0,  0,  0],
            [3,  1,  2,  0,  0,  2,  0,  0,  1],
            [0,  0,  1,  0,  3,  1,  2,  1,  1],
            [2,  0,  1,  0,  0,  1,  2,  1,  1],
            [4,  1,  2,  1,  4,  1,  1,  0,  0],
        ])),
    ]

    game_names = ["mult0", "mult1", "swap", "adder", "compare", "perm", "diff", "cv"]
    

    for i in range(7,8):
        butterfly_game_inequality = butterfly_game_inequalities[i]
        butterfly_facet_inequality = butterfly_facet_inequalities[i]

        print("name = ", game_names[i])
        inequality_tag = "I_" + game_names[i] + "_"

        n_workers = 1
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        # """
        # quantum butterfly game
        # """
        # client.restart()

        # time_start = time.time()

        # qbf_game_opt_fn = optimize_inequality(
        #     [
        #         butterfly_wire_set_nodes,
        #         butterfly_prep_nodes,
        #         butterfly_B_nodes,
        #         butterfly_C_nodes,
        #         butterfly_meas_nodes
        #     ],
        #     np.kron(postmap3,postmap3),
        #     butterfly_game_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # qbf_game_opt_jobs = client.map(qbf_game_opt_fn, range(n_workers))
        # qbf_game_opt_dicts = client.gather(qbf_game_opt_jobs)

        # max_opt_dict = qbf_game_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qbf_game_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qbf_game_opt_dicts[j]["scores"])
        #         max_opt_dict = qbf_game_opt_dicts[j]

        # scenario = "qbf_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # quantum butterfly facet
        # """
        # client.restart()

        # time_start = time.time()

        # qbf_facet_opt_fn = optimize_inequality(
        #     [
        #         butterfly_wire_set_nodes,
        #         butterfly_prep_nodes,
        #         butterfly_B_nodes,
        #         butterfly_C_nodes,
        #         butterfly_meas_nodes
        #     ],
        #     np.kron(postmap3,postmap3),
        #     butterfly_facet_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # qbf_facet_opt_jobs = client.map(qbf_facet_opt_fn, range(n_workers))
        # qbf_facet_opt_dicts = client.gather(qbf_facet_opt_jobs)

        # max_opt_dict = qbf_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qbf_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qbf_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = qbf_facet_opt_dicts[j]

        # scenario = "qbf_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        """
        eatx_quantum butterfly game
        """
        client.restart()

        time_start = time.time()

        eatx_qbf_game_opt_fn = optimize_inequality(
            [
                butterfly_wire_set_nodes,
                eatx_butterfly_source_nodes,
                eatx_butterfly_prep_nodes,
                butterfly_B_nodes,
                butterfly_C_nodes,
                butterfly_meas_nodes
            ],
            np.kron(postmap3,postmap3),
            butterfly_game_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        eatx_qbf_game_opt_jobs = client.map(eatx_qbf_game_opt_fn, range(n_workers))
        eatx_qbf_game_opt_dicts = client.gather(eatx_qbf_game_opt_jobs)

        max_opt_dict = eatx_qbf_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qbf_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qbf_game_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qbf_game_opt_dicts[j]

        scenario = "eatx_qbf_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        eatx quantum butterfly facet
        """
        client.restart()

        time_start = time.time()

        eatx_qbf_facet_opt_fn = optimize_inequality(
            [
                butterfly_wire_set_nodes,
                eatx_butterfly_source_nodes,
                eatx_butterfly_prep_nodes,
                butterfly_B_nodes,
                butterfly_C_nodes,
                butterfly_meas_nodes
            ],
            np.kron(postmap3,postmap3),
            butterfly_facet_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        eatx_qbf_facet_opt_jobs = client.map(eatx_qbf_facet_opt_fn, range(n_workers))
        eatx_qbf_facet_opt_dicts = client.gather(eatx_qbf_facet_opt_jobs)

        max_opt_dict = eatx_qbf_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qbf_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qbf_facet_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qbf_facet_opt_dicts[j]

        scenario = "eatx_qbf_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)