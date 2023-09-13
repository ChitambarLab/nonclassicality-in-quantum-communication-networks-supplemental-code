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

        settings = network_ansatz.rand_network_settings(fixed_setting_ids=fixed_setting_ids, fixed_settings=fixed_settings)
        cost = qnetvo.linear_probs_cost_fn(network_ansatz, inequality[1], postmap)
        opt_dict = _gradient_descent_wrapper(cost, settings, **gradient_kwargs)

        print("\nmax_score : ", max(opt_dict["scores"]))
        print("violation : ", max(opt_dict["scores"]) - inequality[0])

        return opt_dict

    return opt_fn

if __name__=="__main__":


    data_dir = "data/bacon_toner_violations/"

    ghz_prep_node = [
        qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qnetvo.ghz_state),
    ]

    def qubit_measure_nodes(num_in):
        return [
            qnetvo.MeasureNode(
                wires=[i],
                num_in=num_in,
                num_out=2,
                ansatz_fn=qml.ArbitraryUnitary,
                num_settings=3
            ) for i in range(2)
        ]
    

    def qc_ab_prep_nodes(num_in):
        return [
            qnetvo.PrepareNode(num_in=num_in, wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        ]
    
    def qc_ab_proc_nodes(num_in):
        return [
            qnetvo.ProcessingNode(num_in=num_in, wires=[1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        ]
    
    def qc_ab_meas_nodes(num_in):
        return [
            qnetvo.MeasureNode(num_out=2, wires=[0]),
            qnetvo.MeasureNode(num_out=2, wires=[1]),
        ]
    

    ent_cc_ab_wires_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3])
    ]
    ent_cc_ab_source_nodes = [
        qnetvo.PrepareNode(wires=[1,2], ansatz_fn=qnetvo.ghz_state),
    ]

    def ent_cc_ab_prep_nodes(num_in):
        def ent_cc_ab_tx_circ(settings, wires):
            qml.ArbitraryUnitary(settings[0:15], wires=wires[0:2])
            b0 = qml.measure(wires[1])

            return [b0]

        return [
            qnetvo.CCSenderNode(num_in=num_in, wires=[0,1], ansatz_fn=ent_cc_ab_tx_circ, num_settings=15, cc_wires_out=[0]),
        ]

    def ent_cc_ab_rx_nodes(num_in):
        def ent_cc_ab_rx_circ(settings, wires, cc_wires):
            qml.cond(cc_wires[0] == 0, qml.ArbitraryUnitary)(settings[0:15], wires=wires[0:2])
            qml.cond(cc_wires[0] == 1, qml.ArbitraryUnitary)(settings[15:30], wires=wires[0:2])

        return [
            qnetvo.CCReceiverNode(num_in=num_in, wires=[2,3], cc_wires_in=[0], ansatz_fn=ent_cc_ab_rx_circ, num_settings=30),
        ]
    
    ent_cc_ab_meas_nodes = [
        qnetvo.MeasureNode(wires=[0,3])
    ]

    ent_qc_ab_wires_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4])
    ]

    ent_qc_ab_source_nodes = [
        qnetvo.PrepareNode(wires=[2,3], ansatz_fn=qnetvo.ghz_state),
    ]

    def ent_qc_ab_tx_nodes(num_in):
        return [
            qnetvo.ProcessingNode(num_in=num_in, wires=[0,1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        ]

    def ent_qc_ab_rx_nodes(num_in):
        return [
            qnetvo.ProcessingNode(num_in=num_in, wires=[2,3,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        ]
    
    ent_qc_ab_meas_nodes = [
        qnetvo.MeasureNode(wires=[0,4], num_out=4)
    ]


    # signaling d=2 prepare and measure inequalities 
    inequalities = [
        (2, np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])),
        (2, np.array([[1,0,0,0],[0,0,1,0],[1,0,0,1],[0,0,0,1]])),
        (7, np.array([
            [0, 0, 1, 0, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 1, 1, 1],
        ])),
        (13, np.array([
            [1, 2, 0, 2, 1, 2, 0, 2, 1],
            [0, 0, 2, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 2, 0, 0],
            [1, 2, 0, 2, 1, 2, 0, 2, 1],
        ])),
        (2, np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[0,1,1,0]])),
        (2, np.array([[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]])),
        (2, np.array([[1,0,0,0],[0,0,1,0],[0,0,0,1],[0,1,0,0]])),
    ]

    num_in_list = [2, 2, 3, 3, 2, 2, 2]

    for i in range(2,4):
        inequality = inequalities[i]
        num_in = num_in_list[i]

        print("i = ", i)
        inequality_tag = "I_" + str(i) + "_"



        n_workers = 3
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

        # """
        # Entanglement NoSignaling
        # """

        # client.restart()

        # time_start = time.time()

        # qc_opt_fn = optimize_inequality(
        #     [
        #         ghz_prep_node,
        #         qubit_measure_nodes(num_in),
        #     ],
        #     np.eye(4),
        #     inequality,
        #     # fixed_setting_ids=fixed_setting_ry_ids[i],
        #     # fixed_settings=fixed_settings[i],
        #     num_steps=150,
        #     step_size=0.5,
        #     sample_width=1,
        #     verbose=True,
        # )

        # qc_opt_jobs = client.map(qc_opt_fn, range(n_workers))
        # qc_opt_dicts = client.gather(qc_opt_jobs)

        # max_opt_dict = qc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qc_opt_dicts[j]["scores"])
        #         max_opt_dict = qc_opt_dicts[j]

        # scenario = "ent_nosig_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        """
        Entanglement-Assisted Classical Signaling A-> B
        """

        client.restart()

        time_start = time.time()

        qc_opt_fn = optimize_inequality(
            [
                ent_cc_ab_wires_nodes,
                ent_cc_ab_source_nodes,
                ent_cc_ab_prep_nodes(num_in),
                ent_cc_ab_rx_nodes(num_in),
                ent_cc_ab_meas_nodes,
            ],
            np.eye(4),
            inequality,
            num_steps=300,
            step_size=0.2,
            sample_width=1,
            verbose=True,
        )

        qc_opt_jobs = client.map(qc_opt_fn, range(n_workers))
        qc_opt_dicts = client.gather(qc_opt_jobs)

        max_opt_dict = qc_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qc_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qc_opt_dicts[j]["scores"])
                max_opt_dict = qc_opt_dicts[j]

        scenario = "eacc_ab_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)




        """
        Quantum Signaling A -> B
        """
        client.restart()

        time_start = time.time()

        qc_opt_fn = optimize_inequality(
            [
                qc_ab_prep_nodes(num_in),
                qc_ab_proc_nodes(num_in),
                qc_ab_meas_nodes(num_in),
            ],
            np.eye(4),
            inequality,
            # fixed_setting_ids=fixed_setting_ry_ids[i],
            # fixed_settings=fixed_settings[i],
            num_steps=300,
            step_size=0.4,
            sample_width=1,
            verbose=True,
        )

        qc_opt_jobs = client.map(qc_opt_fn, range(n_workers))
        qc_opt_dicts = client.gather(qc_opt_jobs)

        max_opt_dict = qc_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qc_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qc_opt_dicts[j]["scores"])
                max_opt_dict = qc_opt_dicts[j]

        scenario = "qc_ab_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        Entanglement-Assisted Quantum  Signaling A-> B
        """

        client.restart()

        time_start = time.time()

        qc_opt_fn = optimize_inequality(
            [
                ent_qc_ab_wires_nodes,
                ent_qc_ab_source_nodes,
                ent_qc_ab_tx_nodes(num_in),
                ent_qc_ab_rx_nodes(num_in),
                ent_qc_ab_meas_nodes,
            ],
            np.eye(4),
            inequality,
            num_steps=300,
            step_size=0.15,
            sample_width=1,
            verbose=True,
        )

        qc_opt_jobs = client.map(qc_opt_fn, range(n_workers))
        qc_opt_dicts = client.gather(qc_opt_jobs)

        max_opt_dict = qc_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qc_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qc_opt_dicts[j]["scores"])
                max_opt_dict = qc_opt_dicts[j]

        scenario = "eaqc_ab_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)