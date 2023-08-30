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


    data_dir = "data/3-22-33_broadcast_violations/"

    postmap3 = np.array([
        [1,0,0,0],[0,1,0,0],[0,0,1,1],
    ])
    postmap2 = np.array([
        [1,0],[0,1],
    ])


    qbc_wire_set_prep_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3])
    ]
    qbc_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0,2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    qbc_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]


    eacc_bc_wire_set_prep_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6,7,8,9])
    ]

    # eacc_bc_source_nodes = [
    #     qnetvo.PrepareNode(wires=[0,4], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    #     qnetvo.PrepareNode(wires=[2,5], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    # ]
    eacc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,4], ansatz_fn=qnetvo.ghz_state),
        qnetvo.PrepareNode(wires=[2,5], ansatz_fn=qnetvo.ghz_state),
    ]

    def eacc_tx_circ(settings, wires):
        

        qml.RY(settings[0], wires=[wires[2]])
        qml.ctrl(qml.SWAP, (wires[2]))(wires=wires[0:2])

        qml.RY(settings[1], wires=[wires[5]])
        qml.ctrl(qml.SWAP, (wires[5]))(wires=wires[3:5])

        qml.ArbitraryUnitary(settings[2:5], wires=[wires[0]])
        qml.ArbitraryUnitary(settings[5:8], wires=[wires[3]])
        # qml.ArbitraryUnitary(settings[2:17], wires=[wires[0],wires[3]])

        b0 = qml.measure(wires[0])
        b1 = qml.measure(wires[3])

        return [b0, b1]

    eacc_bc_tx_nodes = [
        qnetvo.CCSenderNode(num_in=3, wires=[4,6,7,5,8,9], ansatz_fn=eacc_tx_circ, num_settings=8, cc_wires_out=[0,1]),
    ]

    def eacc_rx_circ(settings, wires, cc_wires):
        qml.cond(
            cc_wires[0] == 0,
            qml.ArbitraryUnitary,
        )(settings[0:15], wires=wires[0:2])

        qml.cond(
            cc_wires[0] == 1,
            qml.ArbitraryUnitary,
        )(settings[15:30], wires=wires[0:2])

    eacc_fixed_setting_ids = [0,1,17,18,34,35]
    eacc_fixed_settings = [0,0,np.pi,np.pi,0,0]


    eacc_bc_rx_nodes = [
        qnetvo.CCReceiverNode(wires=[0,1], ansatz_fn=eacc_rx_circ, num_settings=30, cc_wires_in=[0]),
        qnetvo.CCReceiverNode(wires=[2,3], ansatz_fn=eacc_rx_circ, num_settings=30, cc_wires_in=[1]),
    ]

    eacc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1]),
        qnetvo.MeasureNode(num_out=3, wires=[2,3]),
    ]

    eaqc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qnetvo.ghz_state),
        qnetvo.PrepareNode(wires=[2,3], ansatz_fn=qnetvo.ghz_state),
    ]

    eaqc_bc_prep_nodes = [
        qnetvo.ProcessingNode(num_in=3, wires=[1,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    eaqc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    ghzacc_bc_set_wires = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6])
    ]
    ghzacc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=63)#ansatz_fn=qnetvo.ghz_state),
    ]

    def ghzacc_tx_circ(settings, wires):
        

        qml.ArbitraryUnitary(settings[0:63], wires=wires[0:3])
        
        b0 = qml.measure(wires[1])
        b1 = qml.measure(wires[2])

        return [b0, b1]

    ghzacc_bc_prep_nodes = [
        qnetvo.CCSenderNode(wires=[4,5,6], num_in=3, ansatz_fn=ghzacc_tx_circ, num_settings=63, cc_wires_out=[0,1]),
    ]

    def ghzacc_rx_circ(settings, wires, cc_wires):

        qml.cond(
            cc_wires[0] == 0,
            qml.ArbitraryUnitary,
        )(settings[0:15], wires=wires[0:2])

        qml.cond(
            cc_wires[0] == 1,
            qml.ArbitraryUnitary,
        )(settings[15:30], wires=wires[0:2])
    

    ghzacc_bc_rx_nodes = [
        qnetvo.CCReceiverNode(cc_wires_in=[0], wires=[0,1], ansatz_fn=ghzacc_rx_circ, num_settings=30),
        qnetvo.CCReceiverNode(cc_wires_in=[1], wires=[2,3], ansatz_fn=ghzacc_rx_circ, num_settings=30),
    ]
    ghzacc_bc_meas_nodes = [
        qnetvo.MeasureNode(wires=[0,1], num_out=3),
        qnetvo.MeasureNode(wires=[2,3], num_out=3),
    ]

    earx_qc_bc_set_wires = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5]),
    ]
    earx_qc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2], ansatz_fn=qnetvo.ghz_state),
    ]
    earx_qc_bc_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[1,3], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    earx_qc_bc_proc_nodes = [
        qnetvo.ProcessingNode(wires=[0,1,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        qnetvo.ProcessingNode(wires=[2,3,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),   
    ]
    earx_qc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1]),
        qnetvo.MeasureNode(num_out=3, wires=[2,3])
    ]

    ghzaqc_bc_set_wires=[
        qnetvo.PrepareNode(wires=[0,1,2,3,4])
    ]
    ghzaqc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=63), #ansatz_fn=qnetvo.ghz_state),
    ]
    ghzaqc_bc_prep_nodes = [
        qnetvo.ProcessingNode(num_in=3, wires=[1,3,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    ghzaqc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    onesided_eaqc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qnetvo.ghz_state),
    ]
    onesided_eaqc_bc_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    onesided_eaqc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]


    inequalities = mac.bipartite_broadcast_bounds()

    for i in range(6, len(inequalities)):
        inequality = inequalities[i]
        num_in = inequality[1].shape[1]

        print("i = ", i)
        inequality_tag = "I_" + str(i) + "_"

        n_workers = 3
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

        # """
        # quantum broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # qbc_opt_fn = optimize_inequality(
        #     [
        #         qbc_wire_set_prep_nodes,
        #         qbc_prep_nodes,
        #         qbc_meas_nodes,
        #     ],
        #     np.kron(postmap3,postmap3),
        #     inequality,
        #     num_steps=150,
        #     step_size=0.2,
        #     sample_width=1,
        #     verbose=True
        # )

        # qbc_opt_jobs = client.map(qbc_opt_fn, range(n_workers))
        # qbc_opt_dicts = client.gather(qbc_opt_jobs)

        # max_opt_dict = qbc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qbc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qbc_opt_dicts[j]["scores"])
        #         max_opt_dict = qbc_opt_dicts[j]

        # scenario = "qbc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # entanglement-assisted classcial communication broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # eacc_bc_opt_fn = optimize_inequality(
        #     [
        #         eacc_bc_wire_set_prep_nodes,
        #         eacc_bc_source_nodes,
        #         eacc_bc_tx_nodes,
        #         eacc_bc_rx_nodes,
        #         eacc_bc_meas_nodes,
        #     ],
        #     np.kron(postmap3,postmap3),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=150,
        #     step_size=0.8,
        #     sample_width=1,
        #     verbose=True
        # )

        # eacc_bc_opt_jobs = client.map(eacc_bc_opt_fn, range(n_workers))
        # eacc_bc_opt_dicts = client.gather(eacc_bc_opt_jobs)

        # max_opt_dict = eacc_bc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(eacc_bc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(eacc_bc_opt_dicts[j]["scores"])
        #         max_opt_dict = eacc_bc_opt_dicts[j]

        # scenario = "eacc_bc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # entanglement-assisted quantum communication broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # eaqc_bc_opt_fn = optimize_inequality(
        #     [
        #         eaqc_bc_source_nodes,
        #         eaqc_bc_prep_nodes,
        #         eaqc_bc_meas_nodes,
        #     ],
        #     np.kron(postmap3,postmap3),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=150,
        #     step_size=0.2,
        #     sample_width=1,
        #     verbose=True
        # )

        # eaqc_bc_opt_jobs = client.map(eaqc_bc_opt_fn, range(n_workers))
        # eaqc_bc_opt_dicts = client.gather(eaqc_bc_opt_jobs)

        # max_opt_dict = eaqc_bc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(eaqc_bc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(eaqc_bc_opt_dicts[j]["scores"])
        #         max_opt_dict = eaqc_bc_opt_dicts[j]

        # scenario = "eaqc_bc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # tripartite entanglement-assisted classical communication broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # ghzacc_bc_opt_fn = optimize_inequality(
        #     [
        #         ghzacc_bc_set_wires,
        #         ghzacc_bc_source_nodes,
        #         ghzacc_bc_prep_nodes,
        #         ghzacc_bc_rx_nodes,
        #         ghzacc_bc_meas_nodes,
        #     ],
        #     np.kron(postmap3,postmap3),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=150,
        #     step_size=0.4,
        #     sample_width=1,
        #     verbose=True
        # )

        # ghzacc_bc_opt_jobs = client.map(ghzacc_bc_opt_fn, range(n_workers))
        # ghzacc_bc_opt_dicts = client.gather(ghzacc_bc_opt_jobs)

        # max_opt_dict = ghzacc_bc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(ghzacc_bc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(ghzacc_bc_opt_dicts[j]["scores"])
        #         max_opt_dict = ghzacc_bc_opt_dicts[j]

        # scenario = "ghzacc_bc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # entanglement-assisted receivers quantum communication broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # earx_qc_bc_opt_fn = optimize_inequality(
        #     [
        #         earx_qc_bc_set_wires,
        #         earx_qc_bc_source_nodes,
        #         earx_qc_bc_prep_nodes,
        #         earx_qc_bc_proc_nodes,
        #         earx_qc_bc_meas_nodes
        #     ],
        #     np.kron(postmap3,postmap3),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=150,
        #     step_size=0.4,
        #     sample_width=1,
        #     verbose=True
        # )

        # earx_qc_bc_opt_jobs = client.map(earx_qc_bc_opt_fn, range(n_workers))
        # earx_qc_bc_opt_dicts = client.gather(earx_qc_bc_opt_jobs)

        # max_opt_dict = earx_qc_bc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(earx_qc_bc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(earx_qc_bc_opt_dicts[j]["scores"])
        #         max_opt_dict = earx_qc_bc_opt_dicts[j]

        # scenario = "earx_qc_bc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # tripartite entanglement-assisted quantum communication broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # ghzaqc_bc_opt_fn = optimize_inequality(
        #     [
        #         ghzaqc_bc_set_wires,
        #         ghzaqc_bc_source_nodes,
        #         ghzaqc_bc_prep_nodes,
        #         ghzaqc_bc_meas_nodes
        #     ],
        #     np.kron(postmap3,postmap3),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=250,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # ghzaqc_bc_opt_jobs = client.map(ghzaqc_bc_opt_fn, range(n_workers))
        # ghzaqc_bc_opt_dicts = client.gather(ghzaqc_bc_opt_jobs)

        # max_opt_dict = ghzaqc_bc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(ghzaqc_bc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(ghzaqc_bc_opt_dicts[j]["scores"])
        #         max_opt_dict = ghzaqc_bc_opt_dicts[j]

        # # scenario = "ghzaqc_bc_arb_"
        # scenario = "gea3_bc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

    
        """
        one-sided entanglement-assisted quantum communication broadcast
        """
        client.restart()

        time_start = time.time()

        onesided_eaqc_bc_opt_fn = optimize_inequality(
            [
                onesided_eaqc_bc_source_nodes,
                onesided_eaqc_bc_prep_nodes,
                onesided_eaqc_bc_meas_nodes
            ],
            np.kron(postmap3,postmap3),
            inequality,
            # fixed_setting_ids=eacc_fixed_setting_ids,
            # fixed_settings=eacc_fixed_settings,
            num_steps=150,
            step_size=0.4,
            sample_width=1,
            verbose=True
        )

        onesided_eaqc_bc_opt_jobs = client.map(onesided_eaqc_bc_opt_fn, range(n_workers))
        onesided_eaqc_bc_opt_dicts = client.gather(onesided_eaqc_bc_opt_jobs)

        max_opt_dict = onesided_eaqc_bc_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(onesided_eaqc_bc_opt_dicts[j]["scores"]) > max_score:
                max_score = max(onesided_eaqc_bc_opt_dicts[j]["scores"])
                max_opt_dict = onesided_eaqc_bc_opt_dicts[j]

        scenario = "onesided_eaqc_bc_arb_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)