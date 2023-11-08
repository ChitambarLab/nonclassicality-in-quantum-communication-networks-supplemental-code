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
        opt_dict = qnet.gradient_descent(*opt_args, **opt_kwargs, optimizer="adam")
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



def optimize_inequality(nodes, postmap, inequality,fixed_setting_ids=[], fixed_settings=[], **gradient_kwargs):

    mac_ansatz = qnet.NetworkAnsatz(*nodes)

    def opt_fn(placeholder_param):

        print("\nclassical bound : ", inequality[0])

        settings = mac_ansatz.rand_network_settings(fixed_setting_ids=fixed_setting_ids,fixed_settings=fixed_settings)
        cost = qnet.linear_probs_cost_fn(mac_ansatz, inequality[1], postmap)
        opt_dict = _gradient_descent_wrapper(cost, settings, **gradient_kwargs)

        print("\nmax_score : ", max(opt_dict["scores"]))
        print("violation : ", max(opt_dict["scores"]) - inequality[0])

        return opt_dict


    return opt_fn


if __name__=="__main__":


    data_dir = "data/33-22-3_multiaccess_violations/"


    def qmac_prep_nodes(num_in):
        return [
            qnet.PrepareNode(num_in=num_in, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
            qnet.PrepareNode(num_in=num_in, wires=[1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2)
        ]
    
    def qmac_meas_nodes(num_out):
        return [
            qnet.MeasureNode(num_out=num_out, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15)
        ]

    def qmac_layers(num_in, num_out):
        return [
            qmac_prep_nodes(num_in),
            qmac_meas_nodes(num_out),
        ]

    # qmac_proc3_nodes = [
    #     qnet.ProcessingNode(wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    #     qnet.ProcessingNode(wires=[1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15)
    # ]
    # qmac_meas3_nodes = [
    #     qnet.MeasureNode(num_out=2, wires=[0,2]),
    # ]


    eatx_mac_prep_nodes = [
        qnet.PrepareNode(wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6)
    ]
    def eatx_mac_meas_nodes(num_in):
        return [
            qnet.MeasureNode(
                num_in=num_in, num_out=2, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3
            ),
            qnet.MeasureNode(
                num_in=num_in, num_out=2, wires=[1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3
            )
        ]
    
    def eatx_mac_layers(num_in):
        return [
            eatx_mac_prep_nodes,
            eatx_mac_meas_nodes(num_in),
        ]
    
    def min_eatx_mac_meas_circ(settings, wires):
        qml.RY(settings[0], wires=wires[0:1])
        qml.RZ(settings[1], wires=wires[0:1])

    def min_eatx_mac_meas_nodes(num_in):
        return [
            qnet.MeasureNode(
                num_in=num_in, num_out=2, wires=[0], ansatz_fn=min_eatx_mac_meas_circ, num_settings=2
            ),
            qnet.MeasureNode(
                num_in=num_in, num_out=2, wires=[1], ansatz_fn=min_eatx_mac_meas_circ, num_settings=2
            ),
        ]
    
    def min_eatx_mac_layers(num_in):
        return [
            min_eatx_mac_prep_nodes,
            min_eatx_mac_meas_nodes(num_in),
        ]
    
    min_eatx_mac_prep_nodes = [
        qnet.PrepareNode(wires=[0,1], ansatz_fn=qnet.ghz_state),
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
        # qml.cond(cc_wires[0] == 0, qml.ArbitraryUnitary)(settings[0:3], wires=wires[0:1])
        # qml.cond(cc_wires[1], qml.ArbitraryUnitary)(settings[3:6], wires=wires[0:1])
        qml.cond(cc_wires[0] == 0 and cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[0:15], wires=wires[0:2])
        qml.cond(cc_wires[0] == 0 and cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[15:30], wires=wires[0:2])
        qml.cond(cc_wires[0] == 1 and cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[30:45], wires=wires[0:2])
        qml.cond(cc_wires[0] == 1 and cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[45:60], wires=wires[0:2])

    
    ghza_mac_ccrx_nodes = [
        qnet.CCReceiverNode(wires=[2,7], ansatz_fn=ccrx_circuit, num_settings=60, cc_wires_in=[0,1]),
    ]
    ghza_mac_meas_nodes = [
        qnet.MeasureNode(num_in=1, num_out=2, wires=[7])
    ]

    ghza_mac_no_locc_sender_nodes = [
        qnet.ProcessingNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        qnet.ProcessingNode(num_in=3, wires=[1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        qnet.ProcessingNode(wires=[2], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    ]
    ghza_mac_no_locc_meas_nodes = [
        qnet.MeasureNode(num_in=1, num_out=2, wires=[0,1,2])
    ]


    ea3_qmac_source_nodes = [
        qnet.PrepareNode(wires=[0,1,2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=14),
    ]
    def ea3_cmac_prep_circ(settings, wires):
        qml.ArbitraryUnitary(settings[0:15], wires=wires[0:2])
        return [qml.measure(wires=wires[0:1])]

    def ea3_cmac_prep_nodes(num_in):
        return [
            qnet.CCSenderNode(num_in=num_in,  wires=[0,3], cc_wires_out=[0], ansatz_fn=ea3_cmac_prep_circ, num_settings=15),
            qnet.CCSenderNode(num_in=num_in,  wires=[1,4], cc_wires_out=[1], ansatz_fn=ea3_cmac_prep_circ, num_settings=15),   
        ]

    def ea3_ccrx_circ(settings, wires, cc_wires):
        qml.cond(cc_wires[0] == 0 and cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[0:15], wires=wires[0:2])
        qml.cond(cc_wires[0] == 0 and cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[15:30], wires=wires[0:2])
        qml.cond(cc_wires[0] == 1 and cc_wires[1] == 0, qml.ArbitraryUnitary)(settings[30:45], wires=wires[0:2])
        qml.cond(cc_wires[0] == 1 and cc_wires[1] == 1, qml.ArbitraryUnitary)(settings[45:60], wires=wires[0:2])

    ea3_ccrx_proc_nodes = [
        qnet.CCReceiverNode(wires=[2,5], num_settings=60, ansatz_fn=ea3_ccrx_circ, cc_wires_in=[0,1]),
    ]
    def ea3_cmac_meas_nodes(num_out):
        return [
            qnet.MeasureNode(wires=[2,5], num_out=num_out)
        ]
    
    def ea3_cmac_layers(num_in, num_out):
        return [
            ea3_qmac_source_nodes,
            ea3_cmac_prep_nodes(num_in),
            ea3_ccrx_proc_nodes,
            ea3_cmac_meas_nodes(num_out),
        ]


    def ea3_qmac_proc_nodes(num_in):
        return [
            qnet.ProcessingNode(num_in=num_in, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
            qnet.ProcessingNode(num_in=num_in, wires=[1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        ]
    def ea3_qmac_meas_nodes(num_out):
        return [qnet.MeasureNode(num_out=num_out, wires=[0,1])]
    
    def ea3_qmac_layers(num_in, num_out):
        return [
            ea3_qmac_source_nodes,
            ea3_qmac_proc_nodes(num_in),
            [qnet.ProcessingNode(wires=[0,1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=63)],
            ea3_qmac_meas_nodes(num_out)
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
        qnet.PrepareNode(wires=[0,1], ansatz_fn=qnet.ghz_state),
    ]
    def eatx_qmac_proc_nodes(num_in):
        return [
            qnet.ProcessingNode(num_in=num_in, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
            qnet.ProcessingNode(num_in=num_in, wires=[1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        ]
    
    def eatx_qmac_meas_nodes(num_out):
        return [
            qnet.MeasureNode(num_out=num_out, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15)
        ]

    def eatx_qmac_layers(num_in, num_out):
        return [
            eatx_qmac_prep_nodes,
            eatx_qmac_proc_nodes(num_in),
            eatx_qmac_meas_nodes(num_out),
        ]

    min_eatx_qmac_prep_nodes = [
        qnet.PrepareNode(wires=[0,1], ansatz_fn=qnet.ghz_state),
    ]
    def min_eatx_qmac_prep_circ(settings, wires):
        qml.RY(settings[0],wires=wires[0:1])
        qml.RZ(settings[1],wires=wires[0:1])

    def min_eatx_qmac_proc_nodes(num_in):
        return [
            qnet.ProcessingNode(num_in=num_in, wires=[0], ansatz_fn=min_eatx_qmac_prep_circ, num_settings=2),
            qnet.ProcessingNode(num_in=num_in, wires=[1], ansatz_fn=min_eatx_qmac_prep_circ, num_settings=2),
        ]
    def min_eatx_qmac_meas_circ(settings, wires):
        qml.CNOT(wires=wires[0:2])
        # qml.RY(settings[0],wires=wires[0:1])
        qml.Hadamard(wires=wires[0:1])

    def min_eatx_qmac_meas_nodes(num_out):
        return [
            qnet.MeasureNode(num_out=4, wires=[0,1], ansatz_fn=min_eatx_qmac_meas_circ, num_settings=0)
        ]

    def min_eatx_qmac_layers(num_in, num_out):
        return [
            min_eatx_qmac_prep_nodes,
            min_eatx_qmac_proc_nodes(num_in),
            min_eatx_qmac_meas_nodes(num_out),
        ]



    inequalities = mac.bisender_mac_qubit_simulation_games()
    
    for i in range(0,len(inequalities)):
        inequality = inequalities[i]

        print("i = ", i)
        inequality_tag = "I_" + str(i) + "_"

        num_out = inequality[1].shape[0]
        num_in = int(np.sqrt(inequality[1].shape[1]))

        if num_out == 4:
            postmap = np.eye(4)
        elif num_out == 3:
            postmap = np.array([
                [1,0,0,0],[0,1,1,0],[0,0,0,1],
            ])
        elif num_out == 2:
            postmap = np.array([
                [1,0,0,0],[0,1,1,1],
            ])

        # for postmap_tag in ["and_"]:
        for postmap_tag in ["xor_"]:#
  
            n_workers = 3
            n_jobs = 3
            client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

            # """
            # QMAC
            # """
            # client.restart()
            # time_start = time.time()

            # qmac_opt_fn = optimize_inequality(
            #     qmac_layers(num_in, num_out),
            #     postmap,
            #     inequality,
            #     num_steps=150,
            #     step_size=0.15,
            #     sample_width=1,
            #     verbose=True
            # )

            # qmac_opt_jobs = client.map(qmac_opt_fn, range(n_jobs))
            # qmac_opt_dicts = client.gather(qmac_opt_jobs)

            # max_opt_dict = qmac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,n_jobs):
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
            # QMAC 3-qubit measure node
            # """
            # client.restart()
            # time_start = time.time()

            # qmac_opt_fn = optimize_inequality(
            #     [
            #         qmac_prep_nodes,
            #         qmac_proc3_nodes,
            #         qmac_meas3_nodes,
            #     ],
            #     postmap,
            #     # np.eye(2),
            #     inequality,
            #     num_steps=150,
            #     step_size=0.15,
            #     sample_width=1,
            #     verbose=True
            # )

            # qmac_opt_jobs = client.map(qmac_opt_fn, range(n_jobs))
            # qmac_opt_dicts = client.gather(qmac_opt_jobs)

            # max_opt_dict = qmac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,n_jobs):
            #     if max(qmac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(qmac_opt_dicts[j]["scores"])
            #         max_opt_dict = qmac_opt_dicts[j]

            # scenario = "qmac3_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + datetime_ext,
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
            # EATx CMAC
            # """
            # client.restart()
            # time_start = time.time()

            # ea_mac_opt_fn = optimize_inequality(
            #     eatx_mac_layers(num_in),
            #     postmap,
            #     inequality,
            #     num_steps=150,
            #     step_size=0.15,
            #     sample_width=1,
            #     verbose=True
            # )

            # ea_mac_opt_jobs = client.map(ea_mac_opt_fn, range(n_jobs))
            # ea_mac_opt_dicts = client.gather(ea_mac_opt_jobs)

            # max_opt_dict = ea_mac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,n_jobs):
            #     if max(ea_mac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(ea_mac_opt_dicts[j]["scores"])
            #         max_opt_dict = ea_mac_opt_dicts[j]

            # scenario = "eatx_mac_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            # )

            # print("iteration time  : ", time.time() - time_start)

            # """
            # min EATx CMAC
            # """
            # client.restart()
            # time_start = time.time()

            # ea_mac_opt_fn = optimize_inequality(
            #     min_eatx_mac_layers(num_in),
            #     postmap,
            #     inequality,
            #     num_steps=150,
            #     step_size=0.15,
            #     sample_width=1,
            #     verbose=True
            # )

            # ea_mac_opt_jobs = client.map(ea_mac_opt_fn, range(n_jobs))
            # ea_mac_opt_dicts = client.gather(ea_mac_opt_jobs)

            # max_opt_dict = ea_mac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,n_jobs):
            #     if max(ea_mac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(ea_mac_opt_dicts[j]["scores"])
            #         max_opt_dict = ea_mac_opt_dicts[j]

            # scenario = "min_eatx_mac_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            # )

            # print("iteration time  : ", time.time() - time_start)



            # """
            # EATX QMAC
            # """
            # client.restart()
            # time_start = time.time()

            # ea_qmac_opt_fn = optimize_inequality(
            #     eatx_qmac_layers(num_in,num_out),
            #     postmap,
            #     inequality,
            #     num_steps=160,
            #     step_size=0.12,
            #     sample_width=1,
            #     verbose=True
            # )

            # ea_qmac_opt_jobs = client.map(ea_qmac_opt_fn, range(n_jobs))
            # ea_qmac_opt_dicts = client.gather(ea_qmac_opt_jobs)

            # max_opt_dict = ea_qmac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,n_jobs):
            #     if max(ea_qmac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(ea_qmac_opt_dicts[j]["scores"])
            #         max_opt_dict = ea_qmac_opt_dicts[j]

            # scenario = "eatx_qmac_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            # )

            # print("iteration time  : ", time.time() - time_start)

            # """
            # Min EA QMAC
            
            # optimal settings for 33->22->3 optimal score
            # fixed_setting_ids=[0,1,2,3,4,5,6,7,8,9,10,11],#12],
            # fixed_settings=[0,0,1*np.pi/6,np.pi,np.pi,0,0,0,1*np.pi/6,np.pi,np.pi,0,],#-np.pi/2],
            # """

            # client.restart()
            # time_start = time.time()

            # ea_qmac_opt_fn = optimize_inequality(
            #     min_eatx_qmac_layers(num_in, num_out),
            #     postmap,
            #     inequality,
            #     num_steps=160,
            #     step_size=0.12,
            #     sample_width=1,
            #     verbose=True
            # )

            # ea_qmac_opt_jobs = client.map(ea_qmac_opt_fn, range(n_jobs))
            # ea_qmac_opt_dicts = client.gather(ea_qmac_opt_jobs)

            # max_opt_dict = ea_qmac_opt_dicts[0]
            # max_score = max(max_opt_dict["scores"])
            # for j in range(1,n_jobs):
            #     if max(ea_qmac_opt_dicts[j]["scores"]) > max_score:
            #         max_score = max(ea_qmac_opt_dicts[j]["scores"])
            #         max_opt_dict = ea_qmac_opt_dicts[j]

            # scenario = "min_eatx_qmac_"
            # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            # qnet.write_optimization_json(
            #     max_opt_dict,
            #     data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
            # )

            # print("iteration time  : ", time.time() - time_start)


            # """
            # EA TX RX CMAC
            # """
            # time_start = time.time()

            # ea_txrx_cmac_opt_fn = optimize_inequality(
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

            #     ghza_mac_opt_fn = optimize_inequality(
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
            #         verbose=True
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

            """
            GEA CMAC
            """
            # there is only one qubit being measured.
            if postmap_tag == "xor_":
                time_start = time.time()

                ea3_cmac_opt_fn = optimize_inequality(
                    ea3_cmac_layers(num_in, num_out),
                    postmap,
                    inequality,
                    num_steps=200,
                    step_size=0.08,
                    sample_width=1,
                    verbose=True
                )


                ea3_cmac_opt_jobs = client.map(ea3_cmac_opt_fn, range(n_workers))
                ea3_cmac_opt_dicts = client.gather(ea3_cmac_opt_jobs)

                max_opt_dict = ea3_cmac_opt_dicts[0]
                max_score = max(max_opt_dict["scores"])
                for j in range(1,n_workers):
                    if max(ea3_cmac_opt_dicts[j]["scores"]) > max_score:
                        max_score = max(ea3_cmac_opt_dicts[j]["scores"])
                        max_opt_dict = ea3_cmac_opt_dicts[j]

                scenario = "gea_cmac_"
                datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
                qnet.write_optimization_json(
                    max_opt_dict,
                    data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
                )

                print("iteration time  : ", time.time() - time_start)

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
            #     step_size=0.08,
            #     sample_width=1,
            #     verbose=True
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

            """
            GEA QMAC
            """
            if postmap_tag == "xor_":
                time_start = time.time()

                ea3_qmac_opt_fn = optimize_inequality(
                    ea3_qmac_layers(num_in, num_out),
                    postmap,
                    inequality,
                    num_steps=225,
                    step_size=0.08,
                    sample_width=1,
                    verbose=True,
                )

                ea3_qmac_opt_jobs = client.map(ea3_qmac_opt_fn, range(n_workers))
                ea3_qmac_opt_dicts = client.gather(ea3_qmac_opt_jobs)

                max_opt_dict = ea3_qmac_opt_dicts[0]
                max_score = max(max_opt_dict["scores"])
                for j in range(1,n_workers):
                    if max(ea3_qmac_opt_dicts[j]["scores"]) > max_score:
                        max_score = max(ea3_qmac_opt_dicts[j]["scores"])
                        max_opt_dict = ea3_qmac_opt_dicts[j]

                scenario = "gea_qmac_"
                datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
                qnet.write_optimization_json(
                    max_opt_dict,
                    data_dir + scenario + inequality_tag + postmap_tag + datetime_ext,
                )

                print("iteration time  : ", time.time() - time_start)

            # """
            # GHZA QMAC
            # """
            # if postmap_tag == "xor_":
            #     time_start = time.time()

            #     ea3_qmac_opt_fn = optimize_inequality(
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

                # print("iteration time  : ", time.time() - time_start)