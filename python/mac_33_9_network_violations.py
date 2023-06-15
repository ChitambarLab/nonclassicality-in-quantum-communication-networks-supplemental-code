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


    data_dir = "data/interference_33_33_network_violations/"

    postmap9 = np.array([
        [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    ])

    mac_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3]),
    ]
    mac_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        qnetvo.PrepareNode(num_in=3, wires=[1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
    ]
    mac_meas_nodes = [
        qnetvo.MeasureNode(num_out=9,wires=[0,1,2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]



    mac_game_inequalities = [
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
        (5 , np.array([ # multiplication game [1,2,3] no zero mult1
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
        (5, np.array([ # adder game
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
    
    interference_facet_inequalities = [
        (13, np.array([ # mult zero facet
            [1,  2,  1,  3,  0,  1,  0,  0,  1],
            [1,  0,  0,  1,  3,  0,  0,  1,  2],
            [1,  0,  0,  1,  1,  3,  0,  2,  0],
            [0,  1,  1,  0,  1,  2,  1,  1,  2],
            [0,  1,  1,  0,  1,  2,  1,  1,  2],
            [0,  1,  1,  1,  1,  2,  1,  1,  2],
            [1,  0,  0,  1,  3,  1,  0,  0,  2],
            [1,  1,  0,  1,  1,  2,  0,  1,  2],
            [1,  1,  1,  2,  2,  2,  1,  1,  1],
        ])),
        (14, np.array([ # mult 1 facet
            [3,  0,  1,  0,  2,  1,  1,  0,  1],
            [0,  3,  0,  2,  0,  0,  1,  0,  1],
            [0,  0,  4,  0,  2,  1,  2,  0,  0],
            [2,  1,  2,  0,  2,  1,  0,  1,  1],
            [2,  1,  2,  1,  1,  2,  0,  1,  1],
            [2,  1,  2,  0,  0,  3,  0,  2,  0],
            [2,  1,  2,  0,  2,  1,  0,  0,  1],
            [2,  1,  2,  0,  2,  1,  0,  0,  1],
            [2,  2,  3,  1,  1,  2,  1,  1,  0],
        ])),
        (13, np.array([ # swap facet
            [3,  0,  0,  0,  2,  0,  1,  0,  1],
            [1,  1,  2,  2,  0,  0,  1,  0,  1],
            [0,  1,  3,  0,  1,  1,  2,  0,  0],
            [0,  3,  0,  2,  0,  0,  0,  1,  1],
            [1,  1,  2,  0,  2,  0,  0,  1,  1],
            [1,  0,  3,  1,  0,  1,  0,  2,  0],
            [0,  0,  4,  1,  1,  0,  1,  1,  0],
            [1,  2,  2,  0,  0,  2,  1,  1,  0],
            [2,  2,  3,  1,  1,  1,  1,  1,  0],
        ])),
        (14, np.array([ # adder facet
            [3,  0,  1,  0,  2,  1,  1,  0,  1],
            [0,  3,  0,  2,  0,  0,  1,  0,  1],
            [0,  0,  4,  0,  2,  1,  2,  0,  0],
            [2,  1,  2,  0,  0,  3,  0,  2,  0],
            [1,  0,  3,  1,  1,  2,  1,  1,  1],
            [1,  2,  2,  1,  1,  2,  0,  1,  1],
            [2,  1,  2,  0,  2,  1,  0,  0,  1],
            [1,  1,  3,  0,  2,  1,  1,  0,  1],
            [2,  2,  3,  1,  1,  2,  1,  1,  0],
        ])),
        (12, np.array([ # compare facet
            [2,  0,  0,  0,  3,  0,  0,  0,  1],
            [0,  1,  1,  1,  0,  3,  1,  1,  0],
            [1,  1,  1,  0,  0,  3,  0,  1,  0],
            [0,  2,  1,  0,  2,  2,  1,  0,  1],
            [1,  1,  1,  0,  0,  3,  1,  1,  0],
            [0,  3,  1,  0,  1,  3,  1,  0,  0],
            [1,  2,  1,  1,  1,  2,  0,  0,  1],
            [0,  1,  2,  2,  0,  1,  1,  1,  0],
            [1,  2,  2,  1,  2,  2,  1,  0,  0],
        ])),
        (13, np.array([ # conditioned permutation facet
            [3,  0,  0,  0,  1,  1,  0,  1,  1],
            [0,  3,  2,  1,  0,  1,  0,  0,  1],
            [0,  0,  4,  1,  1,  1,  0,  1,  0],
            [1,  1,  2,  1,  2,  0,  0,  1,  1],
            [1,  2,  2,  1,  0,  3,  0,  1,  0],
            [1,  2,  2,  2,  0,  0,  0,  0,  1],
            [1,  2,  2,  1,  1,  1,  0,  1,  1],
            [0,  1,  3,  0,  1,  2,  1,  0,  0],
            [2,  2,  3,  1,  1,  2,  0,  1,  0],
        ])),
        (11, np.array([ # same difference facet
            [2,  0,  0,  0,  2,  0,  1,  0,  1],
            [0,  0,  2,  1,  1,  2,  1,  0,  1],
            [1,  0,  2,  0,  0,  2,  1,  0,  1],
            [0,  0,  2,  1,  1,  2,  1,  0,  1],
            [0,  0,  2,  2,  0,  2,  0,  1,  0],
            [0,  0,  2,  2,  1,  2,  0,  0,  1],
            [1,  0,  2,  0,  0,  2,  1,  0,  1],
            [0,  0,  2,  2,  1,  2,  0,  0,  1],
            [1,  1,  3,  1,  1,  2,  1,  0,  0],
        ])),
        (13, np.array([ # cv facet
            [3,  0,  0,  0,  2,  0,  1,  0,  1],
            [0,  3,  0,  2,  0,  0,  0,  1,  1],
            [0,  0,  4,  1,  1,  0,  1,  1,  0],
            [1,  1,  2,  2,  0,  0,  1,  0,  1],
            [1,  1,  2,  0,  2,  0,  0,  1,  1],
            [1,  2,  2,  0,  0,  2,  1,  1,  0],
            [0,  1,  3,  0,  1,  1,  2,  0,  0],
            [1,  0,  3,  1,  0,  1,  0,  2,  0],
            [2,  2,  3,  1,  1,  1,  1,  1,  0],
        ])),
    ]

    game_names = ["mult0", "mult1", "swap", "adder", "compare", "perm", "diff", "cv"]
    

    for i in range(0,8):
        interference_game_inequality = interference_game_inequalities[i]
        interference_facet_inequality = interference_facet_inequalities[i]

        print("name = ", game_names[i])
        inequality_tag = "I_" + game_names[i] + "_"

        n_workers = 1
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        # """
        # quantum interference game
        # """
        # client.restart()

        # time_start = time.time()

        # qint_game_opt_fn = optimize_inequality(
        #     [
        #         qint_wire_set_nodes,
        #         qint_prep_nodes,
        #         qint_B_nodes,
        #         qint_meas_nodes,
        #     ],
        #     np.kron(postmap3,postmap3),
        #     interference_game_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # qint_game_opt_jobs = client.map(qint_game_opt_fn, range(n_workers))
        # qint_game_opt_dicts = client.gather(qint_game_opt_jobs)

        # max_opt_dict = qint_game_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qint_game_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qint_game_opt_dicts[j]["scores"])
        #         max_opt_dict = qint_game_opt_dicts[j]

        # scenario = "qint_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # quantum interference facet
        # """
        # client.restart()

        # time_start = time.time()

        # qint_facet_opt_fn = optimize_inequality(
        #     [
        #         qint_wire_set_nodes,
        #         qint_prep_nodes,
        #         qint_B_nodes,
        #         qint_meas_nodes,
        #     ],
        #     np.kron(postmap3,postmap3),
        #     interference_facet_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # qint_facet_opt_jobs = client.map(qint_facet_opt_fn, range(n_workers))
        # qint_facet_opt_dicts = client.gather(qint_facet_opt_jobs)

        # max_opt_dict = qint_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qint_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qint_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = qint_facet_opt_dicts[j]

        # scenario = "qint_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        """
        eatx quantum interference game
        """
        client.restart()

        time_start = time.time()

        print("eatx_qint game")

        eatx_qint_game_opt_fn = optimize_inequality(
            [
                eatx_qint_wire_set_nodes,
                eatx_qint_source_nodes,
                eatx_qint_prep_nodes,
                qint_B_nodes,
                qint_meas_nodes,
            ],
            np.kron(postmap3,postmap3),
            interference_game_inequality,
            num_steps=150,
            step_size=0.15,
            sample_width=1,
            verbose=True
        )

        eatx_qint_game_opt_jobs = client.map(eatx_qint_game_opt_fn, range(n_workers))
        eatx_qint_game_opt_dicts = client.gather(eatx_qint_game_opt_jobs)

        max_opt_dict = eatx_qint_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qint_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qint_game_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qint_game_opt_dicts[j]

        scenario = "eatx_qint_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        eatx quantum interference facet
        """
        client.restart()

        time_start = time.time()

        print("eatx_qint facet")

        eatx_qint_facet_opt_fn = optimize_inequality(
            [
                eatx_qint_wire_set_nodes,
                eatx_qint_source_nodes,
                eatx_qint_prep_nodes,
                qint_B_nodes,
                qint_meas_nodes,
            ],
            np.kron(postmap3,postmap3),
            interference_facet_inequality,
            num_steps=150,
            step_size=0.2,
            sample_width=1,
            verbose=True
        )

        eatx_qint_facet_opt_jobs = client.map(eatx_qint_facet_opt_fn, range(n_workers))
        eatx_qint_facet_opt_dicts = client.gather(eatx_qint_facet_opt_jobs)

        max_opt_dict = eatx_qint_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qint_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qint_facet_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qint_facet_opt_dicts[j]

        scenario = "eatx_qint_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        earx quantum interference game
        """
        client.restart()

        time_start = time.time()

        print("earx_qint game")

        earx_qint_game_opt_fn = optimize_inequality(
            [
                earx_qint_wire_set_nodes,
                earx_qint_source_nodes,
                qint_prep_nodes,
                qint_B_nodes,
                qint_meas_nodes,
            ],
            np.kron(postmap3,postmap3),
            interference_game_inequality,
            num_steps=150,
            step_size=0.15,
            sample_width=1,
            verbose=True
        )

        earx_qint_game_opt_jobs = client.map(earx_qint_game_opt_fn, range(n_workers))
        earx_qint_game_opt_dicts = client.gather(earx_qint_game_opt_jobs)

        max_opt_dict = earx_qint_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(earx_qint_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(earx_qint_game_opt_dicts[j]["scores"])
                max_opt_dict = earx_qint_game_opt_dicts[j]

        scenario = "earx_qint_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        earx quantum interference facet
        """
        client.restart()

        time_start = time.time()

        print("earx facet" )

        earx_qint_facet_opt_fn = optimize_inequality(
            [
                earx_qint_wire_set_nodes,
                earx_qint_source_nodes,
                qint_prep_nodes,
                qint_B_nodes,
                qint_meas_nodes,
            ],
            np.kron(postmap3,postmap3),
            interference_facet_inequality,
            num_steps=150,
            step_size=0.2,
            sample_width=1,
            verbose=True
        )

        earx_qint_facet_opt_jobs = client.map(earx_qint_facet_opt_fn, range(n_workers))
        earx_qint_facet_opt_dicts = client.gather(earx_qint_facet_opt_jobs)

        max_opt_dict = earx_qint_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(earx_qint_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(earx_qint_facet_opt_dicts[j]["scores"])
                max_opt_dict = earx_qint_facet_opt_dicts[j]

        scenario = "earx_qint_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)