import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo


import context
import src

if __name__=="__main__":


    data_dir = "data/butterfly_44_22_network_violations/"

    # postmap3 = np.array([
    #     [1,0,0,0],[0,1,0,0],[0,0,1,1],
    # ])
    # postmap3 = np.array([
    #     [1,0,0,1],[0,1,0,0],[0,0,1,0],
    # ])
    postmap2 = [[1,0,0,1],[0,1,1,0]]

    butterfly_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4])
    ]
    butterfly_prep_nodes = [
        qnetvo.PrepareNode(num_in=4, wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        qnetvo.PrepareNode(num_in=4, wires=[2,4], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    butterfly_B_nodes = [
        qnetvo.ProcessingNode(wires=[1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    butterfly_C_nodes = [
        qnetvo.ProcessingNode(wires=[1,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    butterfly_decoder_nodes = [
        qnetvo.ProcessingNode(wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.ProcessingNode(wires=[3,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    butterfly_meas_nodes = [
        qnetvo.MeasureNode(num_out=2,wires=[0]),
        qnetvo.MeasureNode(num_out=2,wires=[4]),
    ]


    butterfly_game_inequalities = [
        (13, np.array([ # double RAC game
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        ])),
    ]
    
    butterfly_facet_inequalities = [
        (71, np.array([
            [3,  2,  0,  0,  4,  4,   7,   3,  0,  2,  4,  0,  0,  0,  0,  6],
            [2,  0,  6,  8,  0,  2,   0,   0,  0,  1,  0,  8,  0,  2,  2,  0],
            [0,  0,  0,  6,  2,  0,   8,   7,  6,  0,  2,  2,  4,  0,  4,  0],
            [6,  1,  2,  8,  5,  1,  10,  10,  4,  2,  4,  4,  4,  2,  2,  0],
        ])),
    ]

    game_names = ["rac"]
    

    for i in range(0,1):
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

        # qbf_game_opt_fn = src.optimize_inequality(
        #     [
        #         butterfly_wire_set_nodes,
        #         butterfly_prep_nodes,
        #         butterfly_B_nodes,
        #         butterfly_C_nodes,
        #         butterfly_decoder_nodes,
        #         butterfly_meas_nodes
        #     ],
        #     np.kron(np.eye(2),np.eye(2)),
        #     butterfly_game_inequality,
        #     num_steps=200,
        #     step_size=0.2,
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

        """
        quantum butterfly facet
        """
        client.restart()

        time_start = time.time()

        qbf_facet_opt_fn = src.optimize_inequality(
            [
                butterfly_wire_set_nodes,
                butterfly_prep_nodes,
                butterfly_B_nodes,
                butterfly_C_nodes,
                butterfly_decoder_nodes,
                butterfly_meas_nodes,
            ],
            np.kron(np.eye(2),np.eye(2)),
            butterfly_facet_inequality,
            num_steps=200,
            step_size=0.06,
            sample_width=1,
            verbose=True
        )

        qbf_facet_opt_jobs = client.map(qbf_facet_opt_fn, range(n_workers))
        qbf_facet_opt_dicts = client.gather(qbf_facet_opt_jobs)

        max_opt_dict = qbf_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qbf_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qbf_facet_opt_dicts[j]["scores"])
                max_opt_dict = qbf_facet_opt_dicts[j]

        scenario = "qbf_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)