import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo

import context
import src

if __name__=="__main__":


    data_dir = "data/mac_33_9_network_violations/"

    postmap9 = np.array([
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    ])
    postmap9m0 = np.array([
        [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    qmac_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3]),
    ]
    qmac_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        qnetvo.PrepareNode(num_in=3, wires=[1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
    ]
    qmac_meas_nodes = [
        qnetvo.MeasureNode(num_out=9,wires=[0,1,2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=255),
    ]

    qmac_layers = [
        qmac_wire_set_nodes,
        qmac_prep_nodes,
        qmac_meas_nodes,
    ]
    
    eatx_wires_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5]),
    ]
    eatx_qmac_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2], ansatz_fn=qnetvo.ghz_state),
    ]

    eatx_qmac_prep_nodes = [
        qnetvo.ProcessingNode(num_in=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.ProcessingNode(num_in=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    eatx_qmac_meas_nodes = [
        qnetvo.MeasureNode(num_out=9,wires=[0,2,4,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=255),
    ]

    eatx_qmac_layers = [
        eatx_wires_set_nodes,
        eatx_qmac_source_nodes,
        eatx_qmac_prep_nodes,
        eatx_qmac_meas_nodes,
    ]

    ghza_qmac_wire_set_nodes =[
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5]),
    ]
    ghza_qmac_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2,4], ansatz_fn=qnetvo.ghz_state),
    ]
    ghza_qmac_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.PrepareNode(num_in=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    ghza_qmac_meas_nodes = [
        qnetvo.MeasureNode(num_out=9, wires=[0,2,4,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=255),
    ]

    ghza_qmac_layers = [
        ghza_qmac_wire_set_nodes,
        ghza_qmac_source_nodes,
        ghza_qmac_prep_nodes,
        ghza_qmac_meas_nodes,
    ]

    mac_game_inequalities, mac_facet_inequalities, game_names = src.mac_33_22_9_network_bounds()

    for i in range(0,8):
        mac_game_inequality = mac_game_inequalities[i]
        mac_facet_inequality = mac_facet_inequalities[i]

        print("name = ", game_names[i])
        inequality_tag = "I_" + game_names[i] + "_"

        n_workers = 2
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        # """
        # quantum mac game
        # """
        # client.restart()

        # time_start = time.time()

        # qmac_game_opt_fn = src.optimize_inequality(
        #     qmac_layers,
        #     # postmap9,
        #     postmap9m0,
        #     mac_game_inequality,
        #     num_steps=150,
        #     step_size=0.06,
        #     sample_width=1,
        #     verbose=True
        # )

        # qmac_game_opt_jobs = client.map(qmac_game_opt_fn, range(n_workers))
        # qmac_game_opt_dicts = client.gather(qmac_game_opt_jobs)

        # max_opt_dict = qmac_game_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qmac_game_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qmac_game_opt_dicts[j]["scores"])
        #         max_opt_dict = qmac_game_opt_dicts[j]

        # scenario = "qmac_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # quantum mac facet
        # """
        # client.restart()

        # time_start = time.time()

        # qmac_facet_opt_fn = src.optimize_inequality(
        #     qmac_layers,
        #     postmap9,
        #     mac_facet_inequality,
        #     num_steps=150,
        #     step_size=0.06,
        #     sample_width=1,
        #     verbose=True
        # )

        # qmac_facet_opt_jobs = client.map(qmac_facet_opt_fn, range(n_workers))
        # qmac_facet_opt_dicts = client.gather(qmac_facet_opt_jobs)

        # max_opt_dict = qmac_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qmac_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qmac_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = qmac_facet_opt_dicts[j]

        # scenario = "qmac_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        """
        eatx quantum mac game
        """
        client.restart()

        time_start = time.time()

        eatx_qmac_facet_opt_fn = src.optimize_inequality(
            eatx_qmac_layers,
            # postmap9,
            postmap9m0,
            mac_game_inequality,
            num_steps=150,
            step_size=0.06,
            sample_width=1,
            verbose=True
        )

        eatx_qmac_facet_opt_jobs = client.map(eatx_qmac_facet_opt_fn, range(n_workers))
        eatx_qmac_facet_opt_dicts = client.gather(eatx_qmac_facet_opt_jobs)

        max_opt_dict = eatx_qmac_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qmac_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qmac_facet_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qmac_facet_opt_dicts[j]

        scenario = "eatx_qmac_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        eatx quantum mac facet
        """
        client.restart()

        time_start = time.time()

        eatx_qmac_facet_opt_fn = src.optimize_inequality(
            eatx_qmac_layers,
            postmap9,
            mac_facet_inequality,
            num_steps=150,
            step_size=0.06,
            sample_width=1,
            verbose=True
        )

        eatx_qmac_facet_opt_jobs = client.map(eatx_qmac_facet_opt_fn, range(n_workers))
        eatx_qmac_facet_opt_dicts = client.gather(eatx_qmac_facet_opt_jobs)

        max_opt_dict = eatx_qmac_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qmac_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qmac_facet_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qmac_facet_opt_dicts[j]

        scenario = "eatx_qmac_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)


        # """
        # ghz quantum mac game
        # """
        # client.restart()

        # time_start = time.time()

        # ghza_qmac_facet_opt_fn = src.optimize_inequality(
        #     ghza_qmac_layers,
        #     postmap9,
        #     mac_game_inequality,
        #     num_steps=150,
        #     step_size=0.06,
        #     sample_width=1,
        #     verbose=True
        # )

        # ghza_qmac_facet_opt_jobs = client.map(ghza_qmac_facet_opt_fn, range(n_workers))
        # ghza_qmac_facet_opt_dicts = client.gather(ghza_qmac_facet_opt_jobs)

        # max_opt_dict = ghza_qmac_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(ghza_qmac_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(ghza_qmac_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = ghza_qmac_facet_opt_dicts[j]

        # scenario = "ghza_qmac_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # GHZA quantum mac facet
        # """
        # client.restart()

        # time_start = time.time()

        # ghza_qmac_facet_opt_fn = src.optimize_inequality(
        #     ghza_qmac_layers,
        #     postmap9,
        #     mac_facet_inequality,
        #     num_steps=150,
        #     step_size=0.06,
        #     sample_width=1,
        #     verbose=True
        # )

        # ghza_qmac_facet_opt_jobs = client.map(ghza_qmac_facet_opt_fn, range(n_workers))
        # ghza_qmac_facet_opt_dicts = client.gather(ghza_qmac_facet_opt_jobs)

        # max_opt_dict = ghza_qmac_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(ghza_qmac_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(ghza_qmac_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = ghza_qmac_facet_opt_dicts[j]

        # scenario = "ghza_qmac_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)
