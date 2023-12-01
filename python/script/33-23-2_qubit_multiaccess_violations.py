import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo as qnet

import context
import src

"""
The goal of this script is to identify quantum resource configurations of the 3,3->2,3->2 and
3,3->3,2->2 multiaccess networks whose behaviors that can achieve nonclassicality. To achieve this goal,
this script collects numerical optimization data for maximizing nonclassicality against the computed
set facet inequalities for the 3,3->2,3->2 and 3,3->3,2->2 scenarios. Violations of these inequalities
demonstrate that the considered resource configuration requires cannot be simulated classically without
increasing the signaling dimension. Note that the network polytopes considered in these scenarios are
contained by the 3,3->2/3,3/2->2 scenario in which the trit of communication is not linked to one particular
sender.
"""

if __name__=="__main__":


    data_dir = "data/33-23-2_qubit_multiaccess_violations/"


    parity_postmap = np.array([[1,0,0,1],[0,1,1,0]])
    and_postmap = np.array([[1,1,1,0],[0,0,0,1]])
    parity_postmap3 = np.array([
        [1,0,0,1,0,1,1,0],
        [0,1,1,0,1,0,0,1],
    ])
    parity_postmap4 = np.array([
        [1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1],
        [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0],
    ])
    and_postmap3 = np.array([
        [1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,1],
    ])

    qmac_prep_nodes = [
        qnet.PrepareNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        qnet.PrepareNode(num_in=3, wires=[1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2)
    ]
    qmac_meas_nodes = [
        qnet.MeasureNode(num_in=1, num_out=2, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15)
    ]


    eatx_mac_prep_nodes = [
        qnet.PrepareNode(num_in=1, wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6)
    ]
    eatx_mac_meas_nodes = [
        qnet.MeasureNode(
            num_in=3, num_out=2, wires=[0], ansatz_fn=lambda settings, wires: qml.Rot(*settings, wires=wires), num_settings=3
        ),
        qnet.MeasureNode(
            num_in=3, num_out=2, wires=[1], ansatz_fn=lambda settings, wires: qml.Rot(*settings, wires=wires), num_settings=3
        )
    ]

    ghza_mac_prep_nodes = [
        qnet.PrepareNode(num_in=1, wires=[0,1,2], ansatz_fn=qnet.ghz_state),
    ]

    ea3_mac_prep_nodes = [
        qnet.PrepareNode(num_in=1, wires=[0,1,2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=14),
    ]

    def cctx_circuit(settings, wires):
        qml.ArbitraryUnitary(settings[0:3], wires=wires[0:1])
        cc_bit_out = qml.measure(wires[0])
        return [cc_bit_out]
    
    def cctx_swap_circuit(settings, wires):
        # controlled use of entangled state
        qml.RY(settings[0], wires=[wires[2]])
        qml.ctrl(qml.SWAP, (wires[2]))(wires=wires[0:2])
        qml.ArbitraryUnitary(settings[1:4], wires=[wires[0]])
        
        b0 = qml.measure(wires[0])

        return [b0]

    ghza_mac_cctx_nodes = [
        qnet.CCSenderNode(num_in=3, wires=[0], ansatz_fn=cctx_circuit, num_settings=3, cc_wires_out=[0]),
        qnet.CCSenderNode(num_in=3, wires=[1], ansatz_fn=cctx_circuit, num_settings=3, cc_wires_out=[1]),
    ]

    ea3_mac_cctx_nodes = [
        qnet.CCSenderNode(num_in=3, wires=[0,3,4], ansatz_fn=cctx_swap_circuit, num_settings=4, cc_wires_out=[0]),
        qnet.CCSenderNode(num_in=3, wires=[1,5,6], ansatz_fn=cctx_swap_circuit, num_settings=4, cc_wires_out=[1]),
    ]

    def ccrx_circuit(settings, wires, cc_wires):
        # apply quantum operations conditioned on classical communication
        qml.cond(cc_wires[0], qml.ArbitraryUnitary)(settings[0:3], wires=wires[0:1])
        qml.cond(cc_wires[1], qml.ArbitraryUnitary)(settings[3:6], wires=wires[0:1])
    
    ghza_mac_ccrx_nodes = [
        qnet.CCReceiverNode(wires=[2], ansatz_fn=ccrx_circuit, num_settings=6, cc_wires_in=[0,1]),
    ]
    ghza_mac_meas_nodes = [
        qnet.MeasureNode(num_in=1, num_out=2, wires=[2])
    ]

    ghza_mac_no_locc_sender_nodes = [
        qnet.ProcessingNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        qnet.ProcessingNode(num_in=3, wires=[1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        qnet.ProcessingNode(wires=[2], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    ]
    ghza_mac_no_locc_meas_nodes = [
        qnet.MeasureNode(num_in=1, num_out=2, wires=[0,1,2])
    ]


    ghza_qmac_prep_nodes = [
        qnet.PrepareNode(num_in=1, wires=[0,1,2], ansatz_fn=qnet.ghz_state),
    ]
    ghza_qmac_proc_nodes = [
        qnet.ProcessingNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        qnet.ProcessingNode(num_in=3, wires=[1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    ]
    ghza_qmac_meas_nodes = [
        qnet.MeasureNode(num_in=1, num_out=2, wires=[0,1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=63)
    ]


    eatx_qmac_prep_nodes = [
        qnet.PrepareNode(1, [0,1], qml.ArbitraryStatePreparation, 6),
    ]
    eatx_qmac_proc_nodes = [
        qnet.ProcessingNode(3, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3),
        qnet.ProcessingNode(3, [1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3),
    ]
    eatx_qmac_meas_nodes = [
        qnet.MeasureNode(1, 2, [0,1], qml.ArbitraryUnitary, 15)
    ]

    ea_rxtx_cmac_prep_nodes = [
        qnet.PrepareNode(wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        qnet.PrepareNode(wires=[2,3], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]

    ghza_rxtx_cmac_prep_nodes = [
        qnet.PrepareNode(wires=[0,1], ansatz_fn=qnet.ghz_state, num_settings=0),
        qnet.PrepareNode(wires=[2,3], ansatz_fn=qnet.ghz_state, num_settings=0),
    ]


    ea_rxtx_proc_cmac_nodes = [
        qnet.CCSenderNode(num_in=3, wires=[0,4,5], ansatz_fn=cctx_swap_circuit, num_settings=4, cc_wires_out=[0]),
        qnet.CCSenderNode(num_in=3, wires=[2,6,7], ansatz_fn=cctx_swap_circuit, num_settings=4, cc_wires_out=[1]),
    ]


    def ea_rxtx_circuit(settings, wires, cc_wires):
        # apply quantum operations conditioned on classical communication
        # qml.cond(cc_wires[0], qml.ArbitraryUnitary)(settings[0:3], wires=wires[0:1])
        # qml.cond(cc_wires[1], qml.ArbitraryUnitary)(settings[3:6], wires=wires[1:2])
        qml.cond(cc_wires[0] == 0, qml.ArbitraryUnitary)(settings[0:15], wires=[wires[0], wires[2]])
        qml.cond(cc_wires[0] == 1, qml.ArbitraryUnitary)(settings[15:30], wires=[wires[0], wires[2]])
        qml.cond(cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[30:45], wires=[wires[1], wires[3]])
        qml.cond(cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[45:60], wires=[wires[1], wires[3]])
        # qml.cond(cc_wires[0], qml.ArbitraryUnitary)(settings[0:63], wires=wires[0:3])
        # qml.cond(cc_wires[1], qml.ArbitraryUnitary)(settings[63:126], wires=wires[0:3])

        # qml.cond(cc_wires[0] == 0 and cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[0:63], wires=wires[0:3])
        # qml.cond(cc_wires[0] == 0 and cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[63:126], wires=wires[0:3])
        # qml.cond(cc_wires[0] == 1 and cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[126:189], wires=wires[0:3])
        # qml.cond(cc_wires[0] == 1 and cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[189:252], wires=wires[0:3])
       

        def rx_circ(settings, wires):
            # qml.ArbitraryUnitary(settings[0:3], wires=[wires[0]])
            # qml.ArbitraryStatePreparation(settings[3:5], wires=[wires[1]])
            qml.ArbitraryUnitary(settings[0:63], wires=wires[0:2])
            qml.ArbitraryStatePreparation(settings[15:21], wires=wires[2:4])

        # qml.cond(cc_wires[0] == 0 and cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[0:255], wires=wires[0:4])
        # qml.cond(cc_wires[0] == 0 and cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[255:510], wires=wires[0:4])
        # qml.cond(cc_wires[0] == 1 and cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[510:765], wires=wires[0:4])
        # qml.cond(cc_wires[0] == 1 and cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[765:1020], wires=wires[0:4])
        
        # qml.cond(cc_wires[0], rx_circ)(settings[0:21], wires=wires[0:4])
        # qml.cond(cc_wires[1], rx_circ)(settings[21:42], wires=wires[0:4])

        # qml.cond(cc_wires[0]==0, rx_circ)(settings[0:21], wires=[wires[0], wires[2]])
        # qml.cond(cc_wires[0]==1, rx_circ)(settings[6:12], wires=[wires[0], wires[2]])
        # qml.cond(cc_wires[1]==0, rx_circ)(settings[12:18], wires=[wires[1], wires[3]])
        # qml.cond(cc_wires[1]==1, rx_circ)(settings[18:24], wires=[wires[1], wires[3]])

        qml.ArbitraryUnitary(settings[60:315], wires=wires[0:4])


        # qml.ArbitraryUnitary(settings[60:75], wires=wires[0:2])



        # qml.cond(cc_wires[0] == 0 and cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[0:15], wires=wires[0:2])
        # qml.cond(cc_wires[0] == 0 and cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[15:30], wires=wires[0:2])
        # qml.cond(cc_wires[0] == 1 and cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[30:45], wires=wires[0:2])
        # qml.cond(cc_wires[0] == 1 and cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[45:60], wires=wires[0:2])



    
    ea_rxtx_rec_cmac_nodes = [
        qnet.CCReceiverNode(wires=[1,3,8,9], ansatz_fn=ea_rxtx_circuit, num_settings=315, cc_wires_in=[0,1]),
    ]
    ea_rxtx_meas_cmac_nodes = [
        qnet.MeasureNode(num_in=1, num_out=2, wires=[1,3,8,9])
    ]

    inequalities = src.multiaccess_33_23_2_bounds() + src.multiaccess_33_32_2_bounds()
    
    for i in range(0,len(inequalities)):
        inequality = inequalities[i]

        print("i = ", i)
        inequality_tag = "I_" + str(i) + "_"


        for postmap_tag in ["xor_"]:#, "and_"]:
            postmap = parity_postmap if postmap_tag == "xor_" else and_postmap
            postmap3 = parity_postmap3 if postmap_tag == "xor_" else and_postmap3

            n_workers = 3
            n_jobs=3
            client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

            """
            QMAC
            """
            client.restart()
            time_start = time.time()

            qmac_opt_fn = src.optimize_inequality(
                [
                    qmac_prep_nodes,
                    qmac_meas_nodes,
                ],
                postmap,
                inequality,
                num_steps=150,
                step_size=0.15,
                sample_width=1,
                verbose=True
            )

            qmac_opt_jobs = client.map(qmac_opt_fn, range(n_jobs))
            qmac_opt_dicts = client.gather(qmac_opt_jobs)

            max_opt_dict = qmac_opt_dicts[0]
            max_score = max(max_opt_dict["scores"])
            for j in range(1,n_jobs):
                if max(qmac_opt_dicts[j]["scores"]) > max_score:
                    max_score = max(qmac_opt_dicts[j]["scores"])
                    max_opt_dict = qmac_opt_dicts[j]

            scenario = "qmac_"
            datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            qnet.write_optimization_json(
                max_opt_dict,
                data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            )

            print("iteration time  : ", time.time() - time_start)

            # """
            # Bell Measurement QMAC
            # """
            # if postmap_tag == "xor_":
            #     time_start = time.time()

            #     bm_qmac_opt_fn = src.optimize_inequality(
            #         [
            #             qmac_prep_nodes,
            #             bm_qmac_meas_nodes,
            #         ],
            #         np.eye(2),
            #         inequality,
            #         num_steps=150,
            #         step_size=0.15,
            #         sample_width=1,
            #         verbose=False
            #     )

            #     bm_qmac_opt_jobs = client.map(bm_qmac_opt_fn, range(n_workers))
            #     bm_qmac_opt_dicts = client.gather(bm_qmac_opt_jobs)

            #     max_opt_dict = bm_qmac_opt_dicts[0]
            #     max_score = max(max_opt_dict["scores"])
            #     for j in range(1,n_workers):
            #         if max(bm_qmac_opt_dicts[j]["scores"]) > max_score:
            #             max_score = max(bm_qmac_opt_dicts[j]["scores"])
            #             max_opt_dict = bm_qmac_opt_dicts[j]

            #     scenario = "bm2_qmac_"
            #     datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            #     qnet.write_optimization_json(
            #         max_opt_dict,
            #         data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            #     )

            #     print("iteration time  : ", time.time() - time_start)

            # """
            # SWAP Test Measurement QMAC
            # """
            # if postmap_tag == "xor_":
            #     time_start = time.time()

            #     swap_qmac_opt_fn = src.optimize_inequality(
            #         [
            #             qmac_prep_nodes,
            #             swap_qmac_proc_nodes,
            #             swap_qmac_meas_nodes,
            #         ],
            #         np.eye(2),
            #         inequality,
            #         num_steps=150,
            #         step_size=0.15,
            #         sample_width=1,
            #         verbose=False
            #     )

            #     swap_qmac_opt_jobs = client.map(swap_qmac_opt_fn, range(n_workers))
            #     swap_qmac_opt_dicts = client.gather(swap_qmac_opt_jobs)

            #     max_opt_dict = swap_qmac_opt_dicts[0]
            #     max_score = max(max_opt_dict["scores"])
            #     for j in range(1,n_workers):
            #         if max(swap_qmac_opt_dicts[j]["scores"]) > max_score:
            #             max_score = max(swap_qmac_opt_dicts[j]["scores"])
            #             max_opt_dict = swap_qmac_opt_dicts[j]

            #     scenario = "swap_qmac_"
            #     datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            #     qnet.write_optimization_json(
            #         max_opt_dict,
            #         data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            #     )

            #     print("iteration time  : ", time.time() - time_start)

            """
            EATx CMAC
            """
            client.restart()
            time_start = time.time()

            ea_mac_opt_fn = src.optimize_inequality(
                [
                    eatx_mac_prep_nodes,
                    eatx_mac_meas_nodes,
                ],
                postmap,
                inequality,
                num_steps=150,
                step_size=0.15,
                sample_width=1,
                verbose=True
            )

            ea_mac_opt_jobs = client.map(ea_mac_opt_fn, range(n_jobs))
            ea_mac_opt_dicts = client.gather(ea_mac_opt_jobs)

            max_opt_dict = ea_mac_opt_dicts[0]
            max_score = max(max_opt_dict["scores"])
            for j in range(1,n_jobs):
                if max(ea_mac_opt_dicts[j]["scores"]) > max_score:
                    max_score = max(ea_mac_opt_dicts[j]["scores"])
                    max_opt_dict = ea_mac_opt_dicts[j]

            scenario = "eatx_mac_"
            datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            qnet.write_optimization_json(
                max_opt_dict,
                data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            )

            print("iteration time  : ", time.time() - time_start)

            """
            EA QMAC
            """
            client.restart()
            time_start = time.time()

            ea_qmac_opt_fn = src.optimize_inequality(
                [
                    eatx_qmac_prep_nodes,
                    eatx_qmac_proc_nodes,
                    eatx_qmac_meas_nodes,
                ],
                postmap,
                inequality,
                num_steps=160,
                step_size=0.12,
                sample_width=1,
                verbose=True
            )

            ea_qmac_opt_jobs = client.map(ea_qmac_opt_fn, range(n_jobs))
            ea_qmac_opt_dicts = client.gather(ea_qmac_opt_jobs)

            max_opt_dict = ea_qmac_opt_dicts[0]
            max_score = max(max_opt_dict["scores"])
            for j in range(1,n_jobs):
                if max(ea_qmac_opt_dicts[j]["scores"]) > max_score:
                    max_score = max(ea_qmac_opt_dicts[j]["scores"])
                    max_opt_dict = ea_qmac_opt_dicts[j]

            scenario = "eatx_qmac_"
            datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            qnet.write_optimization_json(
                max_opt_dict,
                data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            )

            print("iteration time  : ", time.time() - time_start)

            # """
            # EA TX RX CMAC
            # """
            # time_start = time.time()

            # ea_txrx_cmac_opt_fn = src.optimize_inequality(
            #     [
            #         ghza_rxtx_cmac_prep_nodes,
            #         ea_rxtx_proc_cmac_nodes,
            #         ea_rxtx_rec_cmac_nodes,
            #         ea_rxtx_meas_cmac_nodes,
            #     ],
            #     parity_postmap4,
            #     # np.eye(2),
            #     inequality,
            #     num_steps=170,
            #     step_size=0.2,
            #     sample_width=1,
            #     verbose=True,
            # )

            # ea_txrx_cmac_opt_jobs = client.map(ea_txrx_cmac_opt_fn, range(n_workers))
            # ea_txrx_cmac_opt_dicts = client.gather(ea_txrx_cmac_opt_jobs)

            # max_opt_dict = ea_txrx_cmac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,n_workers):
            #     if max(ea_txrx_cmac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(ea_txrx_cmac_opt_dicts[j]["scores"])
            #         max_opt_dict = ea_txrx_cmac_opt_dicts[j]

            # scenario = "ghza_txrx_cmac_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            # )

            # print("iteration time  : ", time.time() - time_start)

            # """
            # GHZA CMAC
            # """
            # # there is only one qubit being measured.
            # if postmap_tag == "xor_":
            #     time_start = time.time()

            #     ghza_mac_opt_fn = src.optimize_inequality(
            #         [
            #             ghza_mac_prep_nodes,
            #             ghza_mac_cctx_nodes,
            #             ghza_mac_ccrx_nodes,
            #             ghza_mac_meas_nodes,
            #         ],
            #         np.eye(2),
            #         inequality,
            #         num_steps=160,
            #         step_size=0.05,
            #         sample_width=1,
            #         verbose=False
            #     )


            #     ghza_mac_opt_jobs = client.map(ghza_mac_opt_fn, range(n_workers))
            #     ghza_mac_opt_dicts = client.gather(ghza_mac_opt_jobs)

            #     max_opt_dict = ghza_mac_opt_dicts[0]
            #     max_score = max(max_opt_dict["scores"])
            #     for j in range(1,n_workers):
            #         if max(ghza_mac_opt_dicts[j]["scores"]) > max_score:
            #             max_score = max(ghza_mac_opt_dicts[j]["scores"])
            #             max_opt_dict = ghza_mac_opt_dicts[j]

            #     scenario = "ghza_cmac_"
            #     datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            #     qnet.write_optimization_json(
            #         max_opt_dict,
            #         data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            #     )

            #     print("iteration time  : ", time.time() - time_start)

            # """
            # EA3 CMAC
            # """
            # # there is only one qubit being measured.
            # if postmap_tag == "xor_":
            #     time_start = time.time()

            #     ea3_cmac_opt_fn = src.optimize_inequality(
            #         [
            #             ea3_mac_prep_nodes,
            #             ea3_mac_cctx_nodes,
            #             ghza_mac_ccrx_nodes,
            #             ghza_mac_meas_nodes,
            #         ],
            #         np.eye(2),
            #         inequality,
            #         num_steps=160,
            #         step_size=0.05,
            #         sample_width=1,
            #         verbose=False
            #     )


            #     ea3_cmac_opt_jobs = client.map(ea3_cmac_opt_fn, range(n_workers))
            #     ea3_cmac_opt_dicts = client.gather(ea3_cmac_opt_jobs)

            #     max_opt_dict = ea3_cmac_opt_dicts[0]
            #     max_score = max(max_opt_dict["scores"])
            #     for j in range(1,n_workers):
            #         if max(ea3_cmac_opt_dicts[j]["scores"]) > max_score:
            #             max_score = max(ea3_cmac_opt_dicts[j]["scores"])
            #             max_opt_dict = ea3_cmac_opt_dicts[j]

            #     scenario = "ea3_cmac_"
            #     datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            #     qnet.write_optimization_json(
            #         max_opt_dict,
            #         data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            #     )

            #     print("iteration time  : ", time.time() - time_start)

            # """
            # GHZ CMAC no LOCC
            # """
            # # there is only one qubit being measured.
            # time_start = time.time()

            # ghza_mac_opt_fn = src.optimize_inequality(
            #     [
            #         ghza_mac_prep_nodes,
            #         ghza_mac_no_locc_sender_nodes,
            #         ghza_mac_no_locc_meas_nodes,
            #     ],
            #     postmap3,
            #     inequality,
            #     num_steps=160,
            #     step_size=0.1,
            #     sample_width=1,
            #     verbose=False
            # )

            # ghza_mac_opt_jobs = client.map(ghza_mac_opt_fn, range(n_workers))
            # ghza_mac_opt_dicts = client.gather(ghza_mac_opt_jobs)

            # max_opt_dict = ghza_mac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,n_workers):
            #     if max(ghza_mac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(ghza_mac_opt_dicts[j]["scores"])
            #         max_opt_dict = ghza_mac_opt_dicts[j]

            # scenario = "ghza_cmac_no_locc_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            # )

            # print("iteration time  : ", time.time() - time_start)

            # """
            # GHZA QMAC
            # """
            # time_start = time.time()

            # ghza_qmac_opt_fn = src.optimize_inequality(
            #     [
            #         ghza_qmac_prep_nodes,
            #         ghza_qmac_proc_nodes,
            #         ghza_qmac_meas_nodes,
            #     ],
            #     postmap3,
            #     inequality,
            #     num_steps=300,
            #     step_size=0.05,
            #     sample_width=1,
            #     verbose=False
            # )

            # ghza_qmac_opt_jobs = client.map(ghza_qmac_opt_fn, range(n_workers))
            # ghza_qmac_opt_dicts = client.gather(ghza_qmac_opt_jobs)

            # max_opt_dict = ghza_qmac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,n_workers):
            #     if max(ghza_qmac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(ghza_qmac_opt_dicts[j]["scores"])
            #         max_opt_dict = ghza_qmac_opt_dicts[j]

            # scenario = "ghza_qmac_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            # )

            # print("iteration time  : ", time.time() - time_start)

            # """
            # EA3 QMAC
            # """
            # if postmap_tag == "xor_":
            #     time_start = time.time()

            #     ea3_qmac_opt_fn = src.optimize_inequality(
            #         [
            #             ea3_mac_prep_nodes,
            #             ghza_qmac_proc_nodes,
            #             ghza_qmac_meas_nodes,
            #         ],
            #         postmap3,
            #         inequality,
            #         num_steps=200,
            #         step_size=0.05,
            #         sample_width=1,
            #         verbose=False
            #     )

            #     ea3_qmac_opt_jobs = client.map(ea3_qmac_opt_fn, range(n_workers))
            #     ea3_qmac_opt_dicts = client.gather(ea3_qmac_opt_jobs)

            #     max_opt_dict = ea3_qmac_opt_dicts[0]
            #     max_score = max(max_opt_dict["scores"])
            #     for j in range(1,n_workers):
            #         if max(ea3_qmac_opt_dicts[j]["scores"]) > max_score:
            #             max_score = max(ea3_qmac_opt_dicts[j]["scores"])
            #             max_opt_dict = ea3_qmac_opt_dicts[j]

            #     scenario = "ea3_qmac_"
            #     datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            #     qnet.write_optimization_json(
            #         max_opt_dict,
            #         data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            #     )

            #     print("iteration time  : ", time.time() - time_start)

            # """
            # GHZA QMAC
            # """
            # if postmap_tag == "xor_":
            #     time_start = time.time()

            #     ea3_qmac_opt_fn = src.optimize_inequality(
            #         [
            #             ghza_mac_prep_nodes,
            #             ghza_qmac_proc_nodes,
            #             ghza_qmac_meas_nodes,
            #         ],
            #         postmap3,
            #         inequality,
            #         num_steps=250,
            #         step_size=0.08,
            #         sample_width=1,
            #         verbose=False
            #     )

            #     ea3_qmac_opt_jobs = client.map(ea3_qmac_opt_fn, range(n_workers))
            #     ea3_qmac_opt_dicts = client.gather(ea3_qmac_opt_jobs)

            #     max_opt_dict = ea3_qmac_opt_dicts[0]
            #     max_score = max(max_opt_dict["scores"])
            #     for j in range(1,n_workers):
            #         if max(ea3_qmac_opt_dicts[j]["scores"]) > max_score:
            #             max_score = max(ea3_qmac_opt_dicts[j]["scores"])
            #             max_opt_dict = ea3_qmac_opt_dicts[j]

            #     scenario = "ghza_qmac_"
            #     datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            #     qnet.write_optimization_json(
            #         max_opt_dict,
            #         data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            #     )

            #     print("iteration time  : ", time.time() - time_start)