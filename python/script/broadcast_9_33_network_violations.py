import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo

import context
import src

"""
The goal of this script is to identify quantum resource configurations of the 9->2,2->3,3 broadcast
network thatt can produce nonclassical behaviors. To achieve this goal,
this script collects numerical optimization data for maximizing nonclassicality against the computed
set facet inequalities and simulation games for the considered broadcast nettwork. Violations of these
inequalities demonstrate a quantum advanttage.
"""

if __name__=="__main__":


    data_dir = "data/broadcast_9_33_network_violations/"

    postmap3 = np.array([
        [1,0,0,0],[0,1,0,0],[0,0,1,1],
    ])
    postmap3b = np.array([
        [1,0,0,0],[0,1,1,0],[0,0,0,1],
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

    qbc_layers = [
        qbc_wire_set_nodes,
        qbc_prep_nodes,
        qbc_meas_nodes,
    ]

    earx_qbc_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5])
    ]
    earx_qbc_source_nodes = [
        qnetvo.PrepareNode(wires=[1,4], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    earx_qbc_prep_nodes = [
        qnetvo.PrepareNode(num_in=9, wires=[0,3], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    earx_qbc_meas_proc_nodes = [
        qnetvo.ProcessingNode(wires=[0,1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        qnetvo.ProcessingNode(wires=[3,4,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    earx_qbc_meas_nodes = [
        qnetvo.MeasureNode(wires=[0,1], num_out=3),
        qnetvo.MeasureNode(wires=[3,4], num_out=3),
    ]

    earx_qbc_layers = [
        earx_qbc_wire_set_nodes,
        earx_qbc_source_nodes,
        earx_qbc_prep_nodes,
        earx_qbc_meas_proc_nodes,
        earx_qbc_meas_nodes,
    ]

    gea_qbc_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6]),
    ]

    gea_qbc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2,5],ansatz_fn=qml.ArbitraryStatePreparation, num_settings=14),
    ]
    gea_qbc_prep_nodes = [
        qnetvo.ProcessingNode(num_in=9, wires=[0,1,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    gea_qbc_meas_proc_nodes = [
        qnetvo.ProcessingNode(wires=[1,2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        qnetvo.ProcessingNode(wires=[4,5,6], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    gea_qbc_meas_nodes = [
        qnetvo.MeasureNode(wires=[2,3], num_out=3),
        qnetvo.MeasureNode(wires=[5,6], num_out=3),
    ]

    gea_qbc_layers = [
        gea_qbc_wire_set_nodes,
        gea_qbc_source_nodes,
        gea_qbc_prep_nodes,
        gea_qbc_meas_proc_nodes,
        gea_qbc_meas_nodes,
    ]

    bc_game_inequalities, bc_facet_inequalities, game_names = src.broadcast_9_22_33_network_bounds()

    for i in range(6,7):
        bc_game_inequality = bc_game_inequalities[i]
        bc_facet_inequality = bc_facet_inequalities[i]

        print("name = ", game_names[i])
        inequality_tag = "I_" + game_names[i] + "_"

        n_workers = 2
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        """
        quantum broadcast game
        """
        client.restart()

        time_start = time.time()

        qbc_game_opt_fn = src.optimize_inequality(
            qbc_layers,
            np.kron(postmap3b, postmap3b),
            bc_game_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        qbc_game_opt_jobs = client.map(qbc_game_opt_fn, range(n_workers))
        qbc_game_opt_dicts = client.gather(qbc_game_opt_jobs)

        max_opt_dict = qbc_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qbc_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qbc_game_opt_dicts[j]["scores"])
                max_opt_dict = qbc_game_opt_dicts[j]

        scenario = "qbc_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        quantum broadcast facet
        """
        client.restart()

        time_start = time.time()

        qbc_facet_opt_fn = src.optimize_inequality(
            qbc_layers,
            np.kron(postmap3b, postmap3b),
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

        # """
        # earx quantum broadcast game
        # """
        # client.restart()

        # time_start = time.time()

        # earx_qbc_facet_opt_fn = src.optimize_inequality(
        #     earx_qbc_layers,
        #     np.kron(postmap3, postmap3),
        #     bc_game_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # earx_qbc_facet_opt_jobs = client.map(earx_qbc_facet_opt_fn, range(n_workers))
        # earx_qbc_facet_opt_dicts = client.gather(earx_qbc_facet_opt_jobs)

        # max_opt_dict = earx_qbc_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(earx_qbc_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(earx_qbc_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = earx_qbc_facet_opt_dicts[j]

        # scenario = "earx_qbc_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)


        # """
        # earx quantum broadcast facet
        # """
        # client.restart()

        # time_start = time.time()

        # earx_qbc_facet_opt_fn = src.optimize_inequality(
        #     earx_qbc_layers,
        #     np.kron(postmap3, postmap3),
        #     bc_facet_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # earx_qbc_facet_opt_jobs = client.map(earx_qbc_facet_opt_fn, range(n_workers))
        # earx_qbc_facet_opt_dicts = client.gather(earx_qbc_facet_opt_jobs)

        # max_opt_dict = earx_qbc_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(earx_qbc_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(earx_qbc_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = earx_qbc_facet_opt_dicts[j]

        # scenario = "earx_qbc_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # gea quantum broadcast game
        # """
        # client.restart()

        # time_start = time.time()

        # gea_qbc_facet_opt_fn = src.optimize_inequality(
        #     gea_qbc_layers,
        #     np.kron(postmap3, postmap3),
        #     bc_game_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # gea_qbc_facet_opt_jobs = client.map(gea_qbc_facet_opt_fn, range(n_workers))
        # gea_qbc_facet_opt_dicts = client.gather(gea_qbc_facet_opt_jobs)

        # max_opt_dict = gea_qbc_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(gea_qbc_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(gea_qbc_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = gea_qbc_facet_opt_dicts[j]

        # scenario = "gea_qbc_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # gea quantum broadcast facet
        # """
        # client.restart()

        # time_start = time.time()

        # gea_qbc_facet_opt_fn = src.optimize_inequality(
        #     gea_qbc_layers,
        #     np.kron(postmap3, postmap3),
        #     bc_facet_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # gea_qbc_facet_opt_jobs = client.map(gea_qbc_facet_opt_fn, range(n_workers))
        # gea_qbc_facet_opt_dicts = client.gather(gea_qbc_facet_opt_jobs)

        # max_opt_dict = gea_qbc_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(gea_qbc_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(gea_qbc_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = gea_qbc_facet_opt_dicts[j]

        # scenario = "gea_qbc_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)




