import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo

import context
import src

"""
The goal of this script is to identify quantum resource configurations of the 3,3->3,3 butterfly
network that can produce nonclassical behaviors. Note that all communication has signaling dimension d=2.
To achieve this goal, this script collects numerical optimization data for maximizing nonclassicality against the computed
set facet inequalities and simulation games for the considered butterfly nettwork. Violations of these
inequalities demonstrate a quantum advanttage.
"""

if __name__=="__main__":


    data_dir = "data/butterfly_33_33_network_violations/"

    # postmap3 = np.array([
    #     [1,0,0,0],[0,1,0,0],[0,0,1,1],
    # ])
    postmap3 = np.array([
        [1,0,0,1],[0,1,0,0],[0,0,1,0],
    ])
    postmap3b = np.array([
        [1,0,0,0],[0,1,1,0],[0,0,0,1],
    ])
    postmap3c = np.array([
        [1,0,0,0],[0,1,0,0],[0,0,1,1],
    ])
    postmap38 = np.array([
        [1,1,0,0,0,0,0,0],
        [0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1],
    ])

    qbf_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6,7,8])
    ]
    qbf_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0,4], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        qnetvo.PrepareNode(num_in=3, wires=[1,7], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    qbf_B_nodes = [
        qnetvo.ProcessingNode(wires=[0,1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    qbf_C_nodes = [
        qnetvo.ProcessingNode(wires=[2,3,6], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]

    qbf_meas_proc_nodes = [
        qnetvo.ProcessingNode(wires=[3,4,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        qnetvo.ProcessingNode(wires=[6,7,8], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]

    qbf_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[3,4]),
        qnetvo.MeasureNode(num_out=3, wires=[6,7]),
    ]

    qbf_layers = [
        qbf_wire_set_nodes,
        qbf_prep_nodes,
        qbf_B_nodes,
        qbf_C_nodes,
        qbf_meas_proc_nodes,
        qbf_meas_nodes,
    ]

    min_qbf_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4])
    ]

    def min_qbf_prep_circ(settings, wires):
        qml.RY(settings[0], wires=wires[0:1])
        qml.PauliX(wires=wires[1:2])


    min_qbf_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0,2], ansatz_fn=min_qbf_prep_circ, num_settings=1),
        qnetvo.PrepareNode(num_in=3, wires=[1,4], ansatz_fn=min_qbf_prep_circ, num_settings=1),
    ]

    def min_qbf_B_circ(settings, wires):
        qml.CNOT(wires=wires[0:2])

    min_qbf_B_nodes = [
        qnetvo.ProcessingNode(wires=[0,1], ansatz_fn=min_qbf_B_circ, num_settings=0),
    ]
    min_qbf_C_nodes = [
        qnetvo.ProcessingNode(wires=[1,3], ansatz_fn=min_qbf_B_circ, num_settings=0),
    ]

    def min_qbf_meas_circ(settings, wires):
        qml.Hadamard(wires=wires[1:2])
        qml.CNOT(wires=wires[0:2])
        qml.Hadamard(wires=wires[1:2])



    min_qbf_meas_proc_nodes = [
        qnetvo.ProcessingNode(wires=[1,2], ansatz_fn=min_qbf_meas_circ, num_settings=0),
        qnetvo.ProcessingNode(wires=[3,4], ansatz_fn=min_qbf_meas_circ, num_settings=0),
    ]

    min_qbf_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[1,2]),
        qnetvo.MeasureNode(num_out=3, wires=[3,4]),
    ]

    min_qbf_layers = [
        min_qbf_wire_set_nodes,
        min_qbf_prep_nodes,
        min_qbf_B_nodes,
        min_qbf_C_nodes,
        min_qbf_meas_proc_nodes,
        min_qbf_meas_nodes,
    ]


    qbf_cc_wings_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6]),
    ]
    qbf_cc_wings_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0,5], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        qnetvo.PrepareNode(num_in=3, wires=[1,6], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]

    def cc_wings_cctx_circ(settings, wires):

        b0 = qml.measure(wires[0])
        b1 = qml.measure(wires[1])

        return [b0,b1]

    qbf_cc_wings_cctx_nodes = [
        qnetvo.CCSenderNode(wires=[5,6], cc_wires_out=[0,1], ansatz_fn=cc_wings_cctx_circ)
    ]
    qbf_cc_wings_B_nodes = [
        qnetvo.ProcessingNode(wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    qbf_cc_wings_C_nodes = [
        qnetvo.ProcessingNode(wires=[1,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]


    def cc_wings_ccrx_circ(settings, wires, cc_wires):
        qml.cond(cc_wires[0] == 0, qml.ArbitraryUnitary)(settings[0:15], wires=wires[0:2])
        qml.cond(cc_wires[0] == 1, qml.ArbitraryUnitary)(settings[15:30], wires=wires[0:2])

    qbf_cc_wings_ccrx_nodes = [
        qnetvo.CCReceiverNode(cc_wires_in=[0], wires=[1,2], ansatz_fn=cc_wings_ccrx_circ, num_settings=30),
        qnetvo.CCReceiverNode(cc_wires_in=[1], wires=[3,4], ansatz_fn=cc_wings_ccrx_circ, num_settings=30),
    ]

    qbf_cc_wings_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[1,2]),
        qnetvo.MeasureNode(num_out=3, wires=[3,4]),
    ]

    qbf_cc_wings_layers = [
        qbf_cc_wings_wire_set_nodes,
        qbf_cc_wings_prep_nodes,
        qbf_cc_wings_cctx_nodes,
        qbf_cc_wings_B_nodes,
        qbf_cc_wings_C_nodes,
        qbf_cc_wings_ccrx_nodes,
        qbf_cc_wings_meas_nodes,
    ]


    eatx_qbf_wires_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6,7,8,9,10])
    ]
    
    eatx_qbf_source_nodes = [
        qnetvo.PrepareNode(wires=[9,10], ansatz_fn=qnetvo.ghz_state)
    ]
    eatx_qbf_prep_nodes = [
        qnetvo.ProcessingNode(num_in=3, wires=[0,4,9], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        qnetvo.ProcessingNode(num_in=3, wires=[1,7,10], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    eatx_qbf_B_nodes = [
        qnetvo.ProcessingNode(wires=[0,1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    eatx_qbf_C_nodes = [
        qnetvo.ProcessingNode(wires=[2,3,6], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    eatx_qbf_meas_proc_nodes = [
        qnetvo.ProcessingNode(wires=[3,4,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        qnetvo.ProcessingNode(wires=[6,7,8], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    eatx_qbf_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[3,4]),
        qnetvo.MeasureNode(num_out=3, wires=[6,7]),
    ]

    eatx_qbf_layers = [
        eatx_qbf_wires_set_nodes,
        eatx_qbf_source_nodes,
        eatx_qbf_prep_nodes,
        eatx_qbf_B_nodes,
        eatx_qbf_C_nodes,
        eatx_qbf_meas_proc_nodes,
        eatx_qbf_meas_nodes,
    ]

    earx_qbf_wires_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6,7,8])
    ]
    earx_qbf_source_nodes = [
        qnetvo.PrepareNode(wires=[5,8],ansatz_fn = qnetvo.ghz_state),
    ]
    earx_qbf_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0,4], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        qnetvo.PrepareNode(num_in=3, wires=[1,7], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    earx_qbf_B_nodes = [
        qnetvo.ProcessingNode(wires=[0,1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    earx_qbf_C_nodes = [
        qnetvo.ProcessingNode(wires=[2,3,6], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    earx_qbf_rx_nodes = [
        qnetvo.ProcessingNode(wires=[3,4,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        qnetvo.ProcessingNode(wires=[6,7,8], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    earx_qbf_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[3,4]),
        qnetvo.MeasureNode(num_out=3, wires=[6,7]),
    ]

    earx_qbf_layers = [
        earx_qbf_wires_set_nodes,
        earx_qbf_source_nodes,
        earx_qbf_prep_nodes,
        earx_qbf_B_nodes,
        earx_qbf_C_nodes,
        earx_qbf_rx_nodes,
        earx_qbf_meas_nodes,
    ]


    butterfly_game_inequalities, butterfly_facet_inequalities, game_names = src.butterfly_33_33_network_bounds()

    for i in range(6,7):
        butterfly_game_inequality = butterfly_game_inequalities[i]
        butterfly_facet_inequality = butterfly_facet_inequalities[i]

        print("name = ", game_names[i])
        inequality_tag = "I_" + game_names[i] + "_"

        n_workers = 2
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        # """
        # quantum butterfly game
        # """
        # client.restart()

        # time_start = time.time()

        # postmap1 = postmap3
        # postmap2 = postmap3

        # qbf_game_opt_fn = src.optimize_inequality(
        #     qbf_layers,
        #     np.kron(postmap1,postmap2),
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
            
        # max_opt_dict["postmap1"] = postmap1.tolist()
        # max_opt_dict["postmap2"] = postmap2.tolist()


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

        # postmap1 = postmap3
        # postmap2 = postmap3

        # qbf_facet_opt_fn = src.optimize_inequality(
        #     qbf_layers,
        #     np.kron(postmap1,postmap2),
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

        # max_opt_dict["postmap1"] = postmap1.tolist()
        # max_opt_dict["postmap2"] = postmap2.tolist()

        # scenario = "qbf_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        """
        min quantum butterfly game
        """
        client.restart()

        time_start = time.time()

        postmap1 = postmap3
        postmap2 = postmap3


        diff_game_fixed_settings = [np.pi,0,np.pi,0,np.pi,0]


        min_qbf_game_opt_fn = src.optimize_inequality(
            min_qbf_layers,
            np.kron(postmap1,postmap2),
            butterfly_game_inequality,
            fixed_settings = diff_game_fixed_settings,
            fixed_setting_ids = range(len(diff_game_fixed_settings)),
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        min_qbf_game_opt_jobs = client.map(min_qbf_game_opt_fn, range(n_workers))
        min_qbf_game_opt_dicts = client.gather(min_qbf_game_opt_jobs)

        max_opt_dict = min_qbf_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(min_qbf_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(min_qbf_game_opt_dicts[j]["scores"])
                max_opt_dict = min_qbf_game_opt_dicts[j]
            
        max_opt_dict["postmap1"] = postmap1.tolist()
        max_opt_dict["postmap2"] = postmap2.tolist()


        scenario = "min_qbf_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        min quantum butterfly facet
        """
        client.restart()

        time_start = time.time()

        postmap1 = postmap3
        postmap2 = postmap3

        min_qbf_facet_opt_fn = src.optimize_inequality(
            min_qbf_layers,
            np.kron(postmap1,postmap2),
            butterfly_facet_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        min_qbf_facet_opt_jobs = client.map(min_qbf_facet_opt_fn, range(n_workers))
        min_qbf_facet_opt_dicts = client.gather(min_qbf_facet_opt_jobs)

        max_opt_dict = min_qbf_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(min_qbf_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(min_qbf_facet_opt_dicts[j]["scores"])
                max_opt_dict = min_qbf_facet_opt_dicts[j]

        max_opt_dict["postmap1"] = postmap1.tolist()
        max_opt_dict["postmap2"] = postmap2.tolist()

        scenario = "min_qbf_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        # """
        # quantum butterfly cc_wings game
        # """
        # client.restart()

        # time_start = time.time()

        # qbf_game_opt_fn = src.optimize_inequality(
        #     qbf_cc_wings_layers,
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

        # scenario = "qbf_cc_wings_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # quantum butterfly cc wings facet
        # """
        # client.restart()

        # time_start = time.time()

        # qbf_facet_opt_fn = src.optimize_inequality(
        #     qbf_cc_wings_layers,
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

        # scenario = "qbf_cc_wings_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)


        # """
        # eatx_quantum butterfly game
        # """
        # client.restart()

        # time_start = time.time()

        # postmap1 = postmap3
        # postmap2 = postmap3

        # eatx_qbf_game_opt_fn = src.optimize_inequality(
        #     eatx_qbf_layers,
        #     np.kron(postmap1,postmap2),
        #     butterfly_game_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # eatx_qbf_game_opt_jobs = client.map(eatx_qbf_game_opt_fn, range(n_workers))
        # eatx_qbf_game_opt_dicts = client.gather(eatx_qbf_game_opt_jobs)

        # max_opt_dict = eatx_qbf_game_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(eatx_qbf_game_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(eatx_qbf_game_opt_dicts[j]["scores"])
        #         max_opt_dict = eatx_qbf_game_opt_dicts[j]

        # max_opt_dict["postmap1"] = postmap1.tolist()
        # max_opt_dict["postmap2"] = postmap2.tolist()

        # scenario = "eatx_qbf_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # eatx quantum butterfly facet
        # """
        # client.restart()

        # time_start = time.time()

        # postmap1 = postmap3
        # postmap2 = postmap3

        # eatx_qbf_facet_opt_fn = src.optimize_inequality(
        #     eatx_qbf_layers,
        #     np.kron(postmap1,postmap2),
        #     butterfly_facet_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # eatx_qbf_facet_opt_jobs = client.map(eatx_qbf_facet_opt_fn, range(n_workers))
        # eatx_qbf_facet_opt_dicts = client.gather(eatx_qbf_facet_opt_jobs)

        # max_opt_dict = eatx_qbf_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(eatx_qbf_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(eatx_qbf_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = eatx_qbf_facet_opt_dicts[j]

        # max_opt_dict["postmap1"] = postmap1.tolist()
        # max_opt_dict["postmap2"] = postmap2.tolist()

        # scenario = "eatx_qbf_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # earx_quantum butterfly game
        # """
        # client.restart()

        # time_start = time.time()

        # postmap1 = postmap3
        # postmap2 = postmap3

        # earx_qbf_game_opt_fn = src.optimize_inequality(
        #     earx_qbf_layers,
        #     np.kron(postmap1,postmap2),
        #     butterfly_game_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # earx_qbf_game_opt_jobs = client.map(earx_qbf_game_opt_fn, range(n_workers))
        # earx_qbf_game_opt_dicts = client.gather(earx_qbf_game_opt_jobs)

        # max_opt_dict = earx_qbf_game_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(earx_qbf_game_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(earx_qbf_game_opt_dicts[j]["scores"])
        #         max_opt_dict = earx_qbf_game_opt_dicts[j]

        # max_opt_dict["postmap1"] = postmap1.tolist()
        # max_opt_dict["postmap2"] = postmap2.tolist()

        # scenario = "earx_qbf_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # earx quantum butterfly facet
        # """
        # client.restart()

        # time_start = time.time()

        # postmap1 = postmap3
        # postmap2 = postmap3
        
        # earx_qbf_facet_opt_fn = src.optimize_inequality(
        #     earx_qbf_layers,
        #     np.kron(postmap1,postmap2),
        #     butterfly_facet_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # earx_qbf_facet_opt_jobs = client.map(earx_qbf_facet_opt_fn, range(n_workers))
        # earx_qbf_facet_opt_dicts = client.gather(earx_qbf_facet_opt_jobs)

        # max_opt_dict = earx_qbf_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(earx_qbf_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(earx_qbf_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = earx_qbf_facet_opt_dicts[j]
        
        # max_opt_dict["postmap1"] = postmap1.tolist()
        # max_opt_dict["postmap2"] = postmap2.tolist()

        # scenario = "earx_qbf_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)