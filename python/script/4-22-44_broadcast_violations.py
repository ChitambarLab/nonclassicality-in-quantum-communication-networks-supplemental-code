import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo

import context
import src

"""
This script collects numerical optimization data for maximizing nonclassicality in the
4->2,2->4,4 broadcast network against our set of known nonclassicality witnesses for the scenario.
The goal of the script is to identify resource configurations that can provide a quantum nonclassicality
advantage. Notably, we investigate the role of entanglement-assisted receivers in establishing nonclassical
behaviors.
"""

if __name__=="__main__":


    data_dir = "data/4-22-44_broadcast_violations/"

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
        qnetvo.PrepareNode(num_in=4, wires=[0,2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    qbc_meas_nodes = [
        qnetvo.MeasureNode(num_out=4, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=4, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
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
        qnetvo.CCSenderNode(num_in=4, wires=[4,6,7,5,8,9], ansatz_fn=eacc_tx_circ, num_settings=8, cc_wires_out=[0,1]),
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
        qnetvo.MeasureNode(num_out=4, wires=[0,1]),
        qnetvo.MeasureNode(num_out=4, wires=[2,3]),
    ]

    eaqc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qnetvo.ghz_state),
        qnetvo.PrepareNode(wires=[2,3], ansatz_fn=qnetvo.ghz_state),
    ]

    eaqc_bc_prep_nodes = [
        qnetvo.ProcessingNode(num_in=4, wires=[1,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    eaqc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=4, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=4, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    ghzacc_bc_set_wires = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6])
    ]
    ghzacc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2,4], ansatz_fn=qnetvo.ghz_state),
    ]
    geacc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2,4], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=14)#ansatz_fn=qnetvo.ghz_state),
    ]

    def ghzacc_tx_circ(settings, wires):

        qml.ArbitraryUnitary(settings[0:63], wires=wires[0:3])
        
        b0 = qml.measure(wires[1])
        b1 = qml.measure(wires[2])

        return [b0, b1]

    ghzacc_bc_prep_nodes = [
        qnetvo.CCSenderNode(wires=[4,5,6], num_in=4, ansatz_fn=ghzacc_tx_circ, num_settings=63, cc_wires_out=[0,1]),
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
        qnetvo.MeasureNode(wires=[0,1], num_out=4),
        qnetvo.MeasureNode(wires=[2,3], num_out=4),
    ]

    earx_qc_bc_set_wires = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5]),
    ]
    earx_cc_bc_set_wires = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6,7]),
    ]
    earx_qc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2], ansatz_fn=qnetvo.ghz_state),
    ]
    earx_qc_bc_prep_nodes = [
        qnetvo.PrepareNode(num_in=4, wires=[1,3], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    def earx_cc_tx_circ(settings, wires):
        qml.ArbitraryStatePreparation(settings[0:6], wires=wires[0:2])
        # qml.ArbitraryStatePreparation(settings[0:2], wires=[wires[0]])
        # qml.ArbitraryStatePreparation(settings[2:4], wires=[wires[1]])
        # qml.RY(settings[0], wires=[wires[0]])
        # qml.RY(settings[1], wires=[wires[1]])


        b0 = qml.measure(wires[0])
        b1 = qml.measure(wires[1])

        return [b0, b1]


    earx_cc_bc_prep_nodes = [
        qnetvo.CCSenderNode(num_in=4, wires=[4,5], ansatz_fn=earx_cc_tx_circ, num_settings=6, cc_wires_out=[0,1])
    ]
    earx_qc_bc_proc_nodes = [
        qnetvo.ProcessingNode(wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.ProcessingNode(wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),   
    ]

    def earx_cc_rx_circ(settings, wires, cc_wires):
        qml.cond(
            cc_wires[0] == 0,
            qml.ArbitraryUnitary,
        )(settings[0:15], wires=wires[0:2])
        # )(settings[0:63], wires=wires[0:3])

        qml.cond(
            cc_wires[0] == 1,
            qml.ArbitraryUnitary,
        )(settings[15:30], wires=wires[0:2])
        # )(settings[63:126], wires=wires[0:3])

    earx_cc_bc_rx_nodes = [
        qnetvo.CCReceiverNode(wires=[0,1], cc_wires_in=[0], ansatz_fn=earx_cc_rx_circ, num_settings=30),
        qnetvo.CCReceiverNode(wires=[2,3], cc_wires_in=[1], ansatz_fn=earx_cc_rx_circ, num_settings=30),
        # qnetvo.CCReceiverNode(wires=[0,1,6], cc_wires_in=[0], ansatz_fn=earx_cc_rx_circ, num_settings=126),
        # qnetvo.CCReceiverNode(wires=[2,3,7], cc_wires_in=[1], ansatz_fn=earx_cc_rx_circ, num_settings=126),
    ]
    
    earx_qc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=4, wires=[0,1]),
        qnetvo.MeasureNode(num_out=4, wires=[2,3])
    ]

    earx_cc_bc_layers = [
        earx_cc_bc_set_wires,
        earx_qc_bc_source_nodes,
        earx_cc_bc_prep_nodes,
        earx_cc_bc_rx_nodes,
        earx_qc_bc_meas_nodes,
    ]

    ghzaqc_bc_set_wires=[
        qnetvo.PrepareNode(wires=[0,1,2,3,4])
    ]
    ghzaqc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2,4], ansatz_fn=qnetvo.ghz_state),
    ]
    geaqc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2,4], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=14),
    ]
    ghzaqc_bc_prep_nodes = [
        qnetvo.ProcessingNode(num_in=4, wires=[1,3,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    ghzaqc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=4, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=4, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    onesided_eaqc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qnetvo.ghz_state),
    ]
    onesided2_eaqc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[2,3], ansatz_fn=qnetvo.ghz_state),
    ]
    onesided_eaqc_bc_prep_nodes = [
        qnetvo.PrepareNode(num_in=4, wires=[1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    onesided_eaqc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=4, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=4, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    min_earx_qc_bc_set_wires = [
        qnetvo.PrepareNode(wires=[0,1,2,3]),
    ]
    min_earx_qc_bc_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2], ansatz_fn=qnetvo.ghz_state),
    ]
    def min_earx_prep_circ(settings, wires):
        qml.RY(settings[0], wires=wires[0:1])
        qml.RY(settings[1], wires=wires[1:2])



    min_earx_qc_bc_prep_nodes = [
        qnetvo.PrepareNode(num_in=4, wires=[1,3], ansatz_fn=min_earx_prep_circ, num_settings=2),
    ]

    def min_earx_meas_circ(settings, wires):
        qml.CNOT(wires=wires[0:2])
        qml.RY(settings[0], wires=wires[0:1])
        qml.RY(settings[1], wires=wires[1:2])
        qml.CNOT(wires=wires[0:2])
        qml.RY(settings[2], wires=wires[0:1])
        qml.RY(settings[3], wires=wires[1:2])

    min_earx_qc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=4, wires=[0,1], ansatz_fn=min_earx_meas_circ, num_settings=4),
        qnetvo.MeasureNode(num_out=4, wires=[2,3], ansatz_fn=min_earx_meas_circ, num_settings=4),
    ]

    min_earx_qc_layers = [
        min_earx_qc_bc_set_wires,
        min_earx_qc_bc_source_nodes,
        min_earx_qc_bc_prep_nodes,
        min_earx_qc_bc_meas_nodes,
    ]


    def hard_code_prep_circ(settings, wires):
        if settings[0] == 0:
            qml.Hadamard(wires=wires[1])
        elif settings[0] == 1:
            qml.Hadamard(wires=wires[1])
            qml.PauliZ(wires=wires[1])
        elif settings[0] == 2:
            qml.PauliX(wires=wires[0])
            qml.Hadamard(wires=wires[1])
            qml.PauliZ(wires=wires[1])
        elif settings[0] == 3:
            qml.PauliX(wires=wires[0])
            qml.Hadamard(wires=wires[1])
    
    hard_code_earx_qc_bc_prep_nodes = [
        qnetvo.PrepareNode(num_in=4, wires=[1,3], ansatz_fn=hard_code_prep_circ, num_settings=1),
    ]

    def hard_code_meas_a_circ(settings, wires):
        qml.CNOT(wires=wires[0:2])
        # qml.Hadamard(wires=wires[0])
        # qml.PauliZ(wires=wires[0])
        qml.RY(3*np.pi/2, wires=[0])

        qml.RY(np.pi/2, wires=[1])

        # qml.Hadamard(wires=wires[1])
        # qml.Hadamard(wires=wires[1])
        # qml.PauliZ(wires=wires[1])
        
        qml.CNOT(wires=wires[0:2])
        # qml.Hadamard(wires=wires[0])
        qml.RY(np.pi/2, wires=wires[0])
        # qml.RY(3*np.pi/4, wires=[1])
        qml.Hadamard(wires=wires[1])

    def hard_code_meas_b_circ(settings, wires):

        qml.CNOT(wires=wires[0:2])
        qml.RY(settings[0], wires=wires[0])
        
        qml.RY(0,wires=wires[1])

        qml.CNOT(wires=wires[0:2])

        qml.RY(np.pi/4, wires=wires[0])
        qml.RY(3*np.pi/2,wires=wires[1])






    hard_code_earx_qc_bc_meas_nodes = [
        qnetvo.MeasureNode(num_out=4, wires=[0,1], ansatz_fn=hard_code_meas_a_circ, num_settings=0),
        # qnetvo.MeasureNode(num_out=4, wires=[2,3], ansatz_fn=min_earx_meas_circ, num_settings=4),
        qnetvo.MeasureNode(num_out=4, wires=[2,3], ansatz_fn=hard_code_meas_b_circ, num_settings=1),
    ]

    hard_code_earx_qc_layers = [
        min_earx_qc_bc_set_wires,
        min_earx_qc_bc_source_nodes,
        hard_code_earx_qc_bc_prep_nodes,
        hard_code_earx_qc_bc_meas_nodes,
        # min_earx_qc_bc_meas_nodes
    ]


    inequalities = src.broadcast_4_22_44_network_bounds()

    for i in range(1, len(inequalities)-1):
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

        # qbc_opt_fn = src.optimize_inequality(
        #     [
        #         qbc_wire_set_prep_nodes,
        #         qbc_prep_nodes,
        #         qbc_meas_nodes,
        #     ],
        #     np.eye(16),
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

        # eacc_bc_opt_fn = src.optimize_inequality(
        #     # earx_cc_bc_layers,
        #     np.eye(16),
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

        # eaqc_bc_opt_fn = src.optimize_inequality(
        #     [
        #         eaqc_bc_source_nodes,
        #         eaqc_bc_prep_nodes,
        #         eaqc_bc_meas_nodes,
        #     ],
        #     np.eye(16),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=200,
        #     step_size=0.1,
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

        # ghzacc_bc_opt_fn = src.optimize_inequality(
        #     [
        #         ghzacc_bc_set_wires,
        #         geacc_bc_source_nodes,
        #         ghzacc_bc_prep_nodes,
        #         ghzacc_bc_rx_nodes,
        #         ghzacc_bc_meas_nodes,
        #     ],
        #     np.eye(16),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=250,
        #     step_size=0.2,
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

        # scenario = "gea_cc_bc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # tripartite ghz-assisted classical communication broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # ghzacc_bc_opt_fn = src.optimize_inequality(
        #     [
        #         ghzacc_bc_set_wires,
        #         ghzacc_bc_source_nodes,
        #         ghzacc_bc_prep_nodes,
        #         ghzacc_bc_rx_nodes,
        #         ghzacc_bc_meas_nodes,
        #     ],
        #     np.eye(16),
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

        # scenario = "ghza_cc_bc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        """
        entanglement-assisted receivers classical communication broadcast
        """
        client.restart()

        time_start = time.time()

        earx_qc_bc_opt_fn = src.optimize_inequality(
            earx_cc_bc_layers,
            np.eye(16),
            inequality,
            # fixed_setting_ids=eacc_fixed_setting_ids,
            # fixed_settings=eacc_fixed_settings,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        earx_qc_bc_opt_jobs = client.map(earx_qc_bc_opt_fn, range(n_workers))
        earx_qc_bc_opt_dicts = client.gather(earx_qc_bc_opt_jobs)

        max_opt_dict = earx_qc_bc_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(earx_qc_bc_opt_dicts[j]["scores"]) > max_score:
                max_score = max(earx_qc_bc_opt_dicts[j]["scores"])
                max_opt_dict = earx_qc_bc_opt_dicts[j]

        scenario = "earx_cc_bc_arb_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        # """
        # min entanglement-assisted receivers quantum communication broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # fixed_settings = [
        #     0, np.pi/2, 0, 3*np.pi/2, np.pi, 3*np.pi/2, np.pi, np.pi/2,
        #     3*np.pi/2,np.pi/2,np.pi/2,
        #     np.pi/4,
        #     # -2.498091860369874, # optimal to -12
        #     -2.49809186, # optimal to -12
        #     np.pi,
        #     np.pi/4,
        #     np.pi/2
        # ]

        # hardcode_fixed_settings = [0,1,2,3]
        # hardcode_fixed_setting_ids = [0,1,2,3]


        # ansatz = qnetvo.NetworkAnsatz(*min_earx_qc_layers)
        # Pnet = qnetvo.behavior_fn(ansatz)
        # # print(np.array(Pnet(fixed_settings)).round(decimals=8) )
        # print(np.array(Pnet(fixed_settings)))


        # earx_qc_bc_opt_fn = src.optimize_inequality(
        #     min_earx_qc_layers,
        #     # hard_code_earx_qc_layers,
        #     np.eye(16),
        #     inequality,
        #     fixed_setting_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],#,8,9,10,11,12,13,14,15],
        #     fixed_settings=fixed_settings,
        #     # fixed_setting_ids=hardcode_fixed_setting_ids,
        #     # fixed_settings=hardcode_fixed_settings,
        #     num_steps=250,
        #     step_size=0.1,
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

        # scenario = "earx_qc_bc_min_"
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

        # earx_qc_bc_opt_fn = src.optimize_inequality(
        #     [
        #         earx_qc_bc_set_wires,
        #         earx_qc_bc_source_nodes,
        #         earx_qc_bc_prep_nodes,
        #         earx_qc_bc_proc_nodes,
        #         earx_qc_bc_meas_nodes
        #     ],
        #     np.eye(16),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=250,
        #     step_size=0.1,
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
        # tripartite GHZ-assisted quantum communication broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # ghzaqc_bc_opt_fn = src.optimize_inequality(
        #     [
        #         ghzaqc_bc_set_wires,
        #         ghzaqc_bc_source_nodes,
        #         ghzaqc_bc_prep_nodes,
        #         ghzaqc_bc_meas_nodes
        #     ],
        #     np.eye(16),
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

        # scenario = "ghza_qc_bc_arb_"
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

        # ghzaqc_bc_opt_fn = src.optimize_inequality(
        #     [
        #         ghzaqc_bc_set_wires,
        #         geaqc_bc_source_nodes,
        #         ghzaqc_bc_prep_nodes,
        #         ghzaqc_bc_meas_nodes
        #     ],
        #     np.eye(16),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=250,
        #     step_size=0.07,
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

        # scenario = "gea_qc_bc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

    
        # """
        # one-sided entanglement-assisted quantum communication broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # onesided_eaqc_bc_opt_fn = src.optimize_inequality(
        #     [
        #         onesided_eaqc_bc_source_nodes,
        #         onesided_eaqc_bc_prep_nodes,
        #         onesided_eaqc_bc_meas_nodes
        #     ],
        #     np.eye(16),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=150,
        #     step_size=0.2,
        #     sample_width=1,
        #     verbose=True
        # )

        # onesided_eaqc_bc_opt_jobs = client.map(onesided_eaqc_bc_opt_fn, range(n_workers))
        # onesided_eaqc_bc_opt_dicts = client.gather(onesided_eaqc_bc_opt_jobs)

        # max_opt_dict = onesided_eaqc_bc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(onesided_eaqc_bc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(onesided_eaqc_bc_opt_dicts[j]["scores"])
        #         max_opt_dict = onesided_eaqc_bc_opt_dicts[j]

        # scenario = "onesided_eaqc_bc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # one-sided 2 entanglement-assisted quantum communication broadcast
        # """
        # client.restart()

        # time_start = time.time()

        # onesided_eaqc_bc_opt_fn = src.optimize_inequality(
        #     [
        #         onesided2_eaqc_bc_source_nodes,
        #         onesided_eaqc_bc_prep_nodes,
        #         onesided_eaqc_bc_meas_nodes
        #     ],
        #     np.eye(16),
        #     inequality,
        #     # fixed_setting_ids=eacc_fixed_setting_ids,
        #     # fixed_settings=eacc_fixed_settings,
        #     num_steps=150,
        #     step_size=0.2,
        #     sample_width=1,
        #     verbose=True
        # )

        # onesided_eaqc_bc_opt_jobs = client.map(onesided_eaqc_bc_opt_fn, range(n_workers))
        # onesided_eaqc_bc_opt_dicts = client.gather(onesided_eaqc_bc_opt_jobs)

        # max_opt_dict = onesided_eaqc_bc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(onesided_eaqc_bc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(onesided_eaqc_bc_opt_dicts[j]["scores"])
        #         max_opt_dict = onesided_eaqc_bc_opt_dicts[j]

        # scenario = "onesided2_eaqc_bc_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        print("iteration time  : ", time.time() - time_start)