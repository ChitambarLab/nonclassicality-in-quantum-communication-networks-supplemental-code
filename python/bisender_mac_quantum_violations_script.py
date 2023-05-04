import multiple_access_channels as mac
import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo as qnet

def _gradient_descent_wrapper(*opt_args, **opt_kwargs):
    """Wraps ``qnetvo.gradient_descent`` in a try-except block to gracefully
    handle errors during computation.
    This function is called with the same parameters as ``qnetvo.gradient_descent``.
    Optimization errors will result in an empty optimization dictionary.
    """
    try:
        opt_dict = qnet.gradient_descent(*opt_args, **opt_kwargs)
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

def optimize_inequality(nodes, postmap, inequality, **gradient_kwargs):

    mac_ansatz = qnet.NetworkAnsatz(*nodes)

    def opt_fn(placeholder_param):

        print("\nclassical bound : ", inequality[0])

        settings = mac_ansatz.rand_network_settings()
        cost = qnet.linear_probs_cost_fn(mac_ansatz, inequality[1], postmap)
        opt_dict = _gradient_descent_wrapper(cost, settings, **gradient_kwargs)

        print("\nmax_score : ", max(opt_dict["scores"]))
        print("violation : ", max(opt_dict["scores"]) - inequality[0])

        return opt_dict


    return opt_fn


if __name__=="__main__":


    data_dir = "data/bisender_mac_quantum_violations/"


    parity_postmap = np.array([[1,0,0,1],[0,1,1,0]])
    and_postmap = np.array([[1,1,1,0],[0,0,0,1]])
    parity_postmap3 = np.array([
        [1,0,0,1,0,1,1,0],
        [0,1,1,0,1,0,0,1],
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

    bm_qmac_meas_nodes = [
        qnet.MeasureNode(num_in=1, num_out=2, wires=[1], ansatz_fn=qml.adjoint(qnet.ghz_state))
    ]

    def swap_test(settings, wires):
        qml.Hadamard(wires=wires[2])
        qml.ctrl(qml.SWAP, [2])(wires=wires[0:2])
        qml.Hadamard(wires=wires[2])
    
    swap_qmac_proc_nodes = [
        qnet.ProcessingNode(num_in=1, wires=[0,1,2], ansatz_fn=swap_test)     
    ]

    swap_qmac_meas_nodes = [
        qnet.MeasureNode(num_in=1, num_out=2, wires=[2])
    ]

    ea_mac_prep_nodes = [
        qnet.PrepareNode(num_in=1, wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6)
    ]
    ea_mac_meas_nodes = [
        qnet.MeasureNode(
            3, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        ),
        qnet.MeasureNode(
            3, 2, [1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
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

    ghza_mac_cctx_nodes = [
        qnet.CCSenderNode(num_in=3, wires=[0], ansatz_fn=cctx_circuit, num_settings=3, cc_wires_out=[0]),
        qnet.CCSenderNode(num_in=3, wires=[1], ansatz_fn=cctx_circuit, num_settings=3, cc_wires_out=[1]),
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


    ea_qmac_prep_nodes = [
        qnet.PrepareNode(1, [0,1], qml.ArbitraryStatePreparation, 6),
    ]
    ea_qmac_proc_nodes = [
        qnet.ProcessingNode(3, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3),
        qnet.ProcessingNode(3, [1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3),
    ]
    ea_qmac_meas_nodes = [
        qnet.MeasureNode(1, 2, [0,1], qml.ArbitraryUnitary, 15)
    ]

    ea_rxtx_cmac_prep_nodes = [
        qnet.PrepareNode(wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        qnet.PrepareNode(wires=[2,3], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]


    ea_rxtx_proc_cmac_nodes = [
        qnet.CCSenderNode(num_in=3, wires=[0], ansatz_fn=cctx_circuit, num_settings=3, cc_wires_out=[0]),
        qnet.CCSenderNode(num_in=3, wires=[2], ansatz_fn=cctx_circuit, num_settings=3, cc_wires_out=[1]),
    ]


    def ea_rxtx_circuit(settings, wires, cc_wires):
        # apply quantum operations conditioned on classical communication
        qml.cond(cc_wires[0], qml.ArbitraryUnitary)(settings[0:15], wires=wires[0:2])
        qml.cond(cc_wires[1], qml.ArbitraryUnitary)(settings[15:30], wires=wires[0:2])
    
    ea_rxtx_rec_cmac_nodes = [
        qnet.CCReceiverNode(wires=[1,3], ansatz_fn=ea_rxtx_circuit, num_settings=30, cc_wires_in=[0,1]),
    ]
    ea_rxtx_meas_cmac_nodes = [
        qnet.MeasureNode(num_in=1, num_out=4, wires=[1,3])
    ]

    inequalities = [(7, mac.finger_printing_matrix(2,3))] + mac.bisender_mac_bounds()
    
    for i in range(0,20):
        inequality = inequalities[i]

        print("i = ", i)
        inequality_tag = "I_fp_" if i == 0 else "I_" + str(i) + "_"


        for postmap_tag in ["xor_", "and_"]:
            postmap = parity_postmap if postmap_tag == "xor_" else and_postmap
            postmap3 = parity_postmap3 if postmap_tag == "xor_" else and_postmap3

            n_workers = 3
            client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

            # """
            # QMAC
            # """
            # time_start = time.time()

            # qmac_opt_fn = optimize_inequality(
            #     qmac_prep_nodes,
            #     qmac_meas_nodes,
            #     postmap,
            #     inequality,
            #     num_steps=150,
            #     step_size=0.15,
            #     sample_width=1,
            #     verbose=False
            # )

            # qmac_opt_jobs = client.map(qmac_opt_fn, range(5))
            # qmac_opt_dicts = client.gather(qmac_opt_jobs)

            # max_opt_dict = qmac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,5):
            #     if max(qmac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(qmac_opt_dicts[j]["scores"])
            #         max_opt_dict = qmac_opt_dicts[j]

            # scenario = "qmac_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            # )

            # print("iteration time  : ", time.time() - time_start)

            # """
            # Bell Measurement QMAC
            # """
            # if postmap_tag == "xor_":
            #     time_start = time.time()

            #     bm_qmac_opt_fn = optimize_inequality(
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

            #     swap_qmac_opt_fn = optimize_inequality(
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

            # """
            # EA CMAC
            # """
            # time_start = time.time()

            # ea_mac_opt_fn = optimize_inequality(
            #     ea_mac_prep_nodes,
            #     ea_mac_meas_nodes,
            #     postmap,
            #     inequality,
            #     num_steps=150,
            #     step_size=0.15,
            #     sample_width=1,
            #     verbose=False
            # )

            # ea_mac_opt_jobs = client.map(ea_mac_opt_fn, range(5))
            # ea_mac_opt_dicts = client.gather(ea_mac_opt_jobs)

            # max_opt_dict = ea_mac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,5):
            #     if max(ea_mac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(ea_mac_opt_dicts[j]["scores"])
            #         max_opt_dict = ea_mac_opt_dicts[j]

            # scenario = "ea_mac_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            # )

            # print("iteration time  : ", time.time() - time_start)

            # """
            # EA QMAC
            # """
            # time_start = time.time()

            # ea_qmac_opt_fn = optimize_inequality(
            #     ea_qmac_prep_nodes,
            #     ea_qmac_meas_nodes,
            #     postmap,
            #     inequality,
            #     processing_nodes=ea_qmac_proc_nodes,
            #     num_steps=160,
            #     step_size=0.12,
            #     sample_width=1,
            #     verbose=False
            # )

            # ea_qmac_opt_jobs = client.map(ea_qmac_opt_fn, range(5))
            # ea_qmac_opt_dicts = client.gather(ea_qmac_opt_jobs)

            # max_opt_dict = ea_qmac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,5):
            #     if max(ea_qmac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(ea_qmac_opt_dicts[j]["scores"])
            #         max_opt_dict = ea_qmac_opt_dicts[j]

            # scenario = "ea_qmac_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            # )

            # print("iteration time  : ", time.time() - time_start)

            """
            EA QMAC
            """
            time_start = time.time()

            ea_txrx_cmac_opt_fn = optimize_inequality(
                [
                    ea_rxtx_cmac_prep_nodes,
                    ea_rxtx_proc_cmac_nodes,
                    ea_rxtx_rec_cmac_nodes,
                    ea_rxtx_meas_cmac_nodes,
                ],
                postmap,
                inequality,
                processing_nodes=ea_qmac_proc_nodes,
                num_steps=160,
                step_size=0.05,
                sample_width=1,
                verbose=False
            )

            ea_txrx_cmac_opt_jobs = client.map(ea_txrx_cmac_opt_fn, range(n_workers))
            ea_txrx_cmac_opt_dicts = client.gather(ea_txrx_cmac_opt_jobs)

            max_opt_dict = ea_txrx_cmac_opt_dicts[0]
            max_score = max(max_opt_dict["scores"])
            for j in range(1,n_workers):
                if max(ea_txrx_cmac_opt_dicts[j]["scores"]) > max_score:
                    max_score = max(ea_txrx_cmac_opt_dicts[j]["scores"])
                    max_opt_dict = ea_txrx_cmac_opt_dicts[j]

            scenario = "ea_txrx_cmac_"
            datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            qnet.write_optimization_json(
                max_opt_dict,
                data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            )

            print("iteration time  : ", time.time() - time_start)

            # """
            # EA3 CMAC
            # """
            # # there is only one qubit being measured.
            # if postmap_tag == "xor_":
            #     time_start = time.time()

            #     ghza_mac_opt_fn = optimize_inequality(
            #         [
            #             ea3_mac_prep_nodes,
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

            # ghza_mac_opt_fn = optimize_inequality(
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

            # ghza_qmac_opt_fn = optimize_inequality(
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

            #     ea3_qmac_opt_fn = optimize_inequality(
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