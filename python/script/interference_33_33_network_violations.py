import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo

import context
import src


if __name__=="__main__":


    data_dir = "data/interference_33_33_network_violations/"

    postmap3 = np.array([
        [1,0,0,1],[0,1,0,0],[0,0,1,0],
    ])
    postmap3b = np.array([
        [1,0,0,0],[0,1,0,0],[0,0,1,1],
    ])
    postmap2 = np.array([
        [1,1,0,0],[0,0,1,1],[0,0,0,0],
    ])
    postmap23a = np.array([
        [1,0],[0,1],[0,0]
    ])
    postmap23b = np.array([
        [1,0],[0,0],[0,1]
    ])

    qint_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4]),
    ]
    qint_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        qnetvo.PrepareNode(num_in=3, wires=[2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
    ]
    qint_B_nodes = [
        qnetvo.ProcessingNode(wires=[0,2,4,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=255),
    ]

    qint_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        # qnetvo.MeasureNode(num_out=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        # qnetvo.MeasureNode(num_out=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    qint_layers = [
        qint_wire_set_nodes,
        qint_prep_nodes,
        qint_B_nodes,
        qint_meas_nodes,
    ]

    # eatx_qint_wire_set_nodes = [
    #     qnetvo.PrepareNode(wires=[0,1,2,3]),
    # ]
    # eatx_qint_source_nodes = [
    #     qnetvo.PrepareNode(wires=[0,2], ansatz_fn=qnetvo.ghz_state),
    # ]
    # eatx_qint_prep_nodes = [
    #     qnetvo.ProcessingNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3), 
    #     qnetvo.ProcessingNode(num_in=3, wires=[2], ansatz_fn=qml.ArbitraryUnitary, num_settings=3), 
    # ]
    eatx_qint_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6,7]),
    ]
    eatx_qint_source_nodes = [
        qnetvo.PrepareNode(wires=[0,3], ansatz_fn=qnetvo.ghz_state),
    ]
    eatx_qint_prep_nodes = [
        qnetvo.ProcessingNode(num_in=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15), 
        qnetvo.ProcessingNode(num_in=3, wires=[3,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=15), 
    ]
    eatx_qint_proc_nodes = [
        qnetvo.ProcessingNode(wires=[0,3,6,7], ansatz_fn=qml.ArbitraryUnitary, num_settings=255),
    ]
    eatx_qint_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[3,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    eatx_qint_layers = [
        eatx_qint_wire_set_nodes,
        eatx_qint_source_nodes,
        eatx_qint_prep_nodes,
        eatx_qint_proc_nodes,
        eatx_qint_meas_nodes,
    ]

    
    earx_qint_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6,7]),
    ]
    earx_qint_source_nodes = [
        qnetvo.PrepareNode(wires=[1,4], ansatz_fn=qnetvo.ghz_state),
    ]
    earx_qint_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        qnetvo.PrepareNode(num_in=3, wires=[3], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
    ]
    earx_qint_B_nodes = [
        qnetvo.ProcessingNode(wires=[0,3,6,7], ansatz_fn=qml.ArbitraryUnitary, num_settings=255),
    ]
    earx_qint_meas_proc_nodes = [
        qnetvo.ProcessingNode(wires=[0,1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        qnetvo.ProcessingNode(wires=[3,4,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    earx_qint_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1]),
        qnetvo.MeasureNode(num_out=3, wires=[3,4]),
    ]

    earx_qint_layers = [
        earx_qint_wire_set_nodes,
        earx_qint_source_nodes,
        earx_qint_prep_nodes,
        earx_qint_B_nodes,
        earx_qint_meas_proc_nodes,
        earx_qint_meas_nodes,
    ]
    

    # ea3tx_source_nodes = [
    #     qnetvo.PrepareNode(wires=[0,1,2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=14)
    # ]
    # ea3tx_prep_nodes = [
    #     qnetvo.PrepareNode(wires=[0])
    # ]

    
    interference_game_inequalities, interference_facet_inequalities, game_names = src.interference_33_33_network_bounds() 
    for i in range(1,2):
        interference_game_inequality = interference_game_inequalities[i]
        interference_facet_inequality = interference_facet_inequalities[i]

        print("name = ", game_names[i])
        inequality_tag = "I_" + game_names[i] + "_"

        n_workers = 2
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        """
        quantum interference game
        """
        client.restart()

        time_start = time.time()

        classical_postmap = np.array([
            [0,0,0,0],[1,1,0,0],[0,0,1,1]
        ])
        qint_game_opt_fn = src.optimize_inequality(
            qint_layers,
            # np.kron(classical_postmap, classical_postmap),
            np.kron(postmap3,postmap3),
            # np.kron(postmap23a,postmap23b),
            interference_game_inequality,
            num_steps=175,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        qint_game_opt_jobs = client.map(qint_game_opt_fn, range(n_workers))
        qint_game_opt_dicts = client.gather(qint_game_opt_jobs)

        max_opt_dict = qint_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qint_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qint_game_opt_dicts[j]["scores"])
                max_opt_dict = qint_game_opt_dicts[j]

        scenario = "qint_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        quantum interference facet
        """
        client.restart()

        time_start = time.time()

        qint_facet_opt_fn = src.optimize_inequality(
            qint_layers,
            np.kron(postmap3,postmap3),
            interference_facet_inequality,
            num_steps=175,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        qint_facet_opt_jobs = client.map(qint_facet_opt_fn, range(n_workers))
        qint_facet_opt_dicts = client.gather(qint_facet_opt_jobs)

        max_opt_dict = qint_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qint_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qint_facet_opt_dicts[j]["scores"])
                max_opt_dict = qint_facet_opt_dicts[j]

        scenario = "qint_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        # """
        # eatx quantum interference game
        # """
        # client.restart()

        # time_start = time.time()

        # print("eatx_qint game")

        # eatx_qint_game_opt_fn = src.optimize_inequality(
        #     eatx_qint_layers,
        #     np.kron(postmap3,postmap3),
        #     interference_game_inequality,
        #     num_steps=175,
        #     step_size=0.05,
        #     sample_width=1,
        #     verbose=True
        # )

        # eatx_qint_game_opt_jobs = client.map(eatx_qint_game_opt_fn, range(n_workers))
        # eatx_qint_game_opt_dicts = client.gather(eatx_qint_game_opt_jobs)

        # max_opt_dict = eatx_qint_game_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(eatx_qint_game_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(eatx_qint_game_opt_dicts[j]["scores"])
        #         max_opt_dict = eatx_qint_game_opt_dicts[j]

        # scenario = "eatx_qint_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # eatx quantum interference facet
        # """
        # client.restart()

        # time_start = time.time()

        # print("eatx_qint facet")

        # eatx_qint_facet_opt_fn = src.optimize_inequality(
        #     eatx_qint_layers,
        #     np.kron(postmap3,postmap3),
        #     interference_facet_inequality,
        #     num_steps=175,
        #     step_size=0.05,
        #     sample_width=1,
        #     verbose=True
        # )

        # eatx_qint_facet_opt_jobs = client.map(eatx_qint_facet_opt_fn, range(n_workers))
        # eatx_qint_facet_opt_dicts = client.gather(eatx_qint_facet_opt_jobs)

        # max_opt_dict = eatx_qint_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(eatx_qint_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(eatx_qint_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = eatx_qint_facet_opt_dicts[j]

        # scenario = "eatx_qint_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # earx quantum interference game
        # """
        # client.restart()

        # time_start = time.time()

        # print("earx_qint game")

        # earx_qint_game_opt_fn = src.optimize_inequality(
        #     earx_qint_layers,
        #     np.kron(postmap3,postmap3),
        #     interference_game_inequality,
        #     num_steps=175,
        #     step_size=0.15,
        #     sample_width=1,
        #     verbose=True
        # )

        # earx_qint_game_opt_jobs = client.map(earx_qint_game_opt_fn, range(n_workers))
        # earx_qint_game_opt_dicts = client.gather(earx_qint_game_opt_jobs)

        # max_opt_dict = earx_qint_game_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(earx_qint_game_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(earx_qint_game_opt_dicts[j]["scores"])
        #         max_opt_dict = earx_qint_game_opt_dicts[j]

        # scenario = "earx_qint_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # earx quantum interference facet
        # """
        # client.restart()

        # time_start = time.time()

        # print("earx facet" )

        # earx_qint_facet_opt_fn = src.optimize_inequality(
        #     earx_qint_layers,
        #     np.kron(postmap3,postmap3),
        #     interference_facet_inequality,
        #     num_steps=175,
        #     step_size=0.2,
        #     sample_width=1,
        #     verbose=True
        # )

        # earx_qint_facet_opt_jobs = client.map(earx_qint_facet_opt_fn, range(n_workers))
        # earx_qint_facet_opt_dicts = client.gather(earx_qint_facet_opt_jobs)

        # max_opt_dict = earx_qint_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(earx_qint_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(earx_qint_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = earx_qint_facet_opt_dicts[j]

        # scenario = "earx_qint_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)