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


    data_dir = "data/broadcast_9_33_network_violations/"

    postmap3 = np.array([
        [1,0,0,0],[0,1,0,0],[0,0,1,1],
    ])

    qbc_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3]),
    ]
    qbc_prep_nodes = [
        qnetvo.PrepareNode(num_in=9, wires=[0,2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    qbc_meas_nodes = [
        qnetvo.MeasureNode(num_out=3,wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3,wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    earx_qbc_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3])
    ]
    earx_qbc_source_nodes = [
        qnetvo.PrepareNode(wires=[1,3], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    earx_qbc_prep_nodes = [
        qnetvo.PrepareNode(num_in=9, wires=[0,2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    earx_qbc_meas_nodes = [
        qnetvo.MeasureNode(num_out=3,wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3,wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]



    bc_game_inequalities = [
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
        (6, np.array([ # multiplication game [1,2,3] no zero mult1
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
        (6, np.array([ # compare game
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
        (4, np.array([ # one receiver permutes output based on other receiver
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
        (7, np.array([ # same difference game
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
        (4, np.eye(9)), # communication value
    ]
    
    bc_facet_inequalities = [
        (5, np.array([ # mult zero face
            [1,  0,  0,  1,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  2,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  2,  0,  0,  0],
            [1,  0,  0,  0,  0,  0,  0,  0,  1],
            [0,  0,  0,  0,  1,  0,  0,  0,  1],
            [0,  0,  0,  0,  0,  2,  0,  0,  1],
            [1,  0,  0,  0,  1,  1,  0,  0,  0],
            [1,  0,  0,  0,  1,  1,  0,  0,  0],
            [1,  0,  0,  0,  1,  2,  0,  0,  0],
        ])),
        (5, np.array([ # mult 1 facet
            [2,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  2,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  2,  0,  0,  0,  0,  0,  0],
            [1,  0,  0,  0,  0,  1,  0,  0,  0],
            [0,  1,  0,  0,  0,  1,  0,  0,  0],
            [0,  0,  2,  0,  0,  1,  0,  0,  0],
            [1,  1,  1,  0,  0,  0,  0,  0,  0],
            [1,  1,  1,  0,  0,  0,  0,  0,  0],
            [1,  1,  2,  0,  0,  0,  0,  0,  0],
        ])),
        (6, np.array([ # swap facet
            [2,  0,  0,  0,  0,  0,  0,  0,  0],
            [1,  0,  0,  1,  0,  0,  0,  0,  0],
            [1,  1,  2,  0,  0,  0,  0,  0,  0],
            [0,  2,  0,  0,  0,  0,  0,  0,  0],
            [0,  1,  0,  0,  1,  0,  0,  0,  0],
            [1,  1,  2,  0,  0,  0,  0,  0,  0],
            [0,  0,  3,  0,  0,  0,  0,  0,  0],
            [0,  0,  3,  0,  0,  0,  0,  0,  0],
            [1,  1,  3,  0,  0,  0,  0,  0,  0],
        ])),
        (6, np.array([ # adder facet
            [2,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  1,  0,  1,  0,  0,  0,  0,  0],
            [0,  0,  3,  0,  0,  0,  0,  0,  0],
            [1,  0,  0,  0,  0,  1,  0,  0,  0],
            [0,  1,  0,  0,  0,  0,  0,  0,  1],
            [0,  0,  3,  0,  0,  0,  0,  0,  0],
            [1,  1,  2,  0,  0,  0,  0,  0,  0],
            [1,  1,  2,  0,  0,  0,  0,  0,  0],
            [1,  1,  3,  0,  0,  0,  0,  0,  0],
        ])),
        (4, np.array([ # compare facet
            [1,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  2,  0,  0,  0,  0,  0],
            [0,  2,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  1],
            [0,  0,  0,  2,  0,  0,  0,  0,  0],
            [0,  2,  0,  0,  0,  0,  0,  0,  0],
            [0,  1,  0,  1,  0,  0,  0,  0,  0],
            [0,  1,  0,  2,  0,  0,  0,  0,  0],
            [0,  2,  0,  1,  0,  0,  0,  0,  0],
        ])),
        (6, np.array([ # conditioned permutation facet
            [2,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  2,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  3,  0,  0,  0,  0,  0,  0],
            [1,  0,  0,  0,  1,  0,  0,  0,  0],
            [0,  1,  0,  0,  0,  1,  0,  0,  0],
            [0,  0,  3,  0,  0,  0,  0,  0,  0],
            [1,  1,  2,  0,  0,  0,  0,  0,  0],
            [1,  1,  2,  0,  0,  0,  0,  0,  0],
            [1,  1,  3,  0,  0,  0,  0,  0,  0],
        ])),
        (5, np.array([ # same difference facet
            [1,  0,  0,  0,  1,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  1],
            [0,  0,  2,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  1],
            [0,  2,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  2,  0,  0,  0,  0,  0,  0],
            [0,  0,  2,  0,  0,  0,  0,  0,  0],
            [0,  0,  2,  0,  0,  0,  0,  0,  0],
            [1,  1,  2,  0,  0,  0,  0,  0,  0],
        ])),
        (6, np.array([ # cv facet
            [2,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  2,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  3,  0,  0,  0,  0,  0,  0],
            [1,  0,  0,  1,  0,  0,  0,  0,  0],
            [0,  1,  0,  0,  1,  0,  0,  0,  0],
            [0,  0,  3,  0,  0,  0,  0,  0,  0],
            [1,  1,  2,  0,  0,  0,  0,  0,  0],
            [1,  1,  2,  0,  0,  0,  0,  0,  0],
            [1,  1,  3,  0,  0,  0,  0,  0,  0],
        ])),
    ]

    game_names = ["mult0", "mult1", "swap", "adder", "compare", "perm", "diff", "cv"]
    

    for i in range(0,8):
        bc_game_inequality = bc_game_inequalities[i]
        bc_facet_inequality = bc_facet_inequalities[i]

        print("name = ", game_names[i])
        inequality_tag = "I_" + game_names[i] + "_"

        n_workers = 1
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        # """
        # quantum broadcast game
        # """
        # client.restart()

        # time_start = time.time()

        # qbc_game_opt_fn = optimize_inequality(
        #     [
        #         qbc_wire_set_nodes,
        #         qbc_prep_nodes,
        #         qbc_meas_nodes,
        #     ],
        #     np.kron(postmap3, postmap3),
        #     bc_game_inequality,
        #     num_steps=100,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # qbc_game_opt_jobs = client.map(qbc_game_opt_fn, range(n_workers))
        # qbc_game_opt_dicts = client.gather(qbc_game_opt_jobs)

        # max_opt_dict = qbc_game_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qbc_game_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qbc_game_opt_dicts[j]["scores"])
        #         max_opt_dict = qbc_game_opt_dicts[j]

        # scenario = "qbc_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        """
        quantum broadcast facet
        """
        client.restart()

        time_start = time.time()

        qbc_facet_opt_fn = optimize_inequality(
            [
                qbc_wire_set_nodes,
                qbc_prep_nodes,
                qbc_meas_nodes,
            ],
            np.kron(postmap3, postmap3),
            bc_facet_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        qbc_facet_opt_jobs = client.map(qbc_facet_opt_fn, range(n_workers))
        qbc_facet_opt_dicts = client.gather(qbc_facet_opt_jobs)

        max_opt_dict = qbc_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qbc_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qbc_facet_opt_dicts[j]["scores"])
                max_opt_dict = qbc_facet_opt_dicts[j]

        scenario = "qbc_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        earx quantum broadcast facet
        """
        client.restart()

        time_start = time.time()

        earx_qbc_facet_opt_fn = optimize_inequality(
            [
                earx_qbc_wire_set_nodes,
                earx_qbc_source_nodes,
                earx_qbc_prep_nodes,
                earx_qbc_meas_nodes,
            ],
            np.kron(postmap3, postmap3),
            bc_facet_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        earx_qbc_facet_opt_jobs = client.map(earx_qbc_facet_opt_fn, range(n_workers))
        earx_qbc_facet_opt_dicts = client.gather(earx_qbc_facet_opt_jobs)

        max_opt_dict = earx_qbc_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(earx_qbc_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(earx_qbc_facet_opt_dicts[j]["scores"])
                max_opt_dict = earx_qbc_facet_opt_dicts[j]

        scenario = "earx_qbc_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)


