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


    data_dir = "data/prepare_and_measure_violations/"

    postmap = np.eye(2)
    parity_postmap = np.array([
        [1,0,0,1],[0,1,1,0],
    ])
    parity_postmap3 = np.array([
        [1,0,0,1,0,1,1,0],[0,1,1,0,1,0,0,1],
    ])

    ghz_prep_node = [
        qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qnetvo.ghz_state),
    ]
    twoghz_prep_nodes = [
        qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qnetvo.ghz_state),
        qnetvo.PrepareNode(wires=[2,3], ansatz_fn=qnetvo.ghz_state),
    ]
    arb_prep_node = [
        qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]


    def eacc_arb_tx(settings, wires):
        # controlled use of entangled state
        qml.RY(settings[0], wires=[wires[2]])
        qml.ctrl(qml.SWAP, (wires[2]))(wires=wires[0:2])
        qml.ArbitraryUnitary(settings[1:4], wires=[wires[0]])
        
        b0 = qml.measure(wires[0])

        return [b0]
    
    def eacc_ry_tx(settings, wires):
        # controlled use of entangled state
        qml.RY(settings[0], wires=[wires[2]])
        qml.ctrl(qml.SWAP, (wires[2]))(wires=wires[0:2])
        qml.RY(settings[1], wires=[wires[0]])
        
        b0 = qml.measure(wires[0])

        return [b0]

    def qc_tx_nodes(num_in, ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2):
        return [
            qnetvo.PrepareNode(
                wires=[0],
                num_in=num_in,
                ansatz_fn=ansatz_fn,
                num_settings=num_settings,
            ),
        ]

    def eacc_tx_nodes(num_in, ansatz_fn=eacc_arb_tx, num_settings=4):
        return [
            qnetvo.CCSenderNode(
                wires=[0,2,3],
                cc_wires_out=[0],
                num_in=num_in,
                ansatz_fn=ansatz_fn,
                num_settings=num_settings,
            ),
        ]
    
    def eaqc_tx_nodes(num_in, ansatz_fn=qml.ArbitraryUnitary, num_settings=3):
        return [
            qnetvo.ProcessingNode(
                wires=[0],
                num_in=num_in,
                ansatz_fn=ansatz_fn,
                num_settings=num_settings,
            ),
        ]
    
    def eaqc2_tx_nodes(num_in, ansatz_fn=qml.ArbitraryUnitary, num_settings=15):
        return [
            qnetvo.ProcessingNode(
                wires=[0,2],
                num_in=num_in,
                ansatz_fn=ansatz_fn,
                num_settings=num_settings,
            ),
        ]

    def eacc_arb_rx(settings, wires, cc_wires):

        qml.cond(
            cc_wires[0] == 0,
            qml.ArbitraryUnitary,
        )(settings[0:15], wires=wires[0:2])

        # qml.cond(cc_wires[0] == 1, qml.SWAP)(wires=wires[0:2])
        qml.cond(
            cc_wires[0] == 1,
            qml.ArbitraryUnitary,
        )(settings[15:30], wires=wires[0:2])


    def eacc_rx_nodes(num_in, ansatz_fn=eacc_arb_rx, num_settings=30):
        return [
            qnetvo.CCReceiverNode(
                wires=[1,4],
                cc_wires_in=[0],
                num_in=num_in,
                num_settings=num_settings,
                ansatz_fn=ansatz_fn,
            ),
        ]
    
    def eaqc_rx_nodes(num_in, ansatz_fn=qml.ArbitraryUnitary, num_settings=15):
        return [
            qnetvo.MeasureNode(
                wires=[0,1],
                num_in=num_in,
                num_out=2,
                num_settings=num_settings,
                ansatz_fn=ansatz_fn,
            ),
        ]
    
    def eaqc2_rx_nodes(num_in, ansatz_fn=qml.ArbitraryUnitary, num_settings=63):
        return [
            qnetvo.MeasureNode(
                wires=[0,1,3],
                num_in=num_in,
                num_out=2,
                num_settings=num_settings,
                ansatz_fn=ansatz_fn,
            ),
        ]

    # eacc_arb_rx_node = [
    #     qnetvo.CCReceiverNode(
    #         wires=[1,4],
    #         cc_wires_in=[0],
    #         num_in=2,
    #         num_settings=30,
    #         ansatz_fn=eacc_arb_rx,
    #     )
    #

    def qc_rx_nodes(num_in, ansatz_fn=qml.ArbitraryUnitary, num_settings=3):
        return [
            qnetvo.MeasureNode(
                wires=[0],
                num_in=num_in,
                num_out=2,
                num_settings=num_settings,
                ansatz_fn=ansatz_fn,
            ),
        ]

    qubit_measure_node = [
        qnetvo.MeasureNode(
            num_out=2,
            wires=[1],
        ),
    ]
    twoqubit_measure_node = [
        qnetvo.MeasureNode(
            num_out=2,
            wires=[1,4],
        )
    ]

    # eaqc_arb_rx_node = [
    #     qnetvo.MeasureNode(
    #         wires=[0,1],
    #         num_in=2,
    #         num_out=2,
    #         num_settings=15,
    #         ansatz_fn=qml.ArbitraryUnitary,
    #     ),
    # ]


    # signaling d=2 prepare and measure inequalities 
    inequalities = [
        (4, np.array([ # 3 input dimensionality witness
            [0,  0,  0,  1,  1,  0],
            [1,  1,  1,  0,  0,  0],
        ])),
        (7, np.array([  # finger printing 33-23-2 tight 
            [1,0,0,0,1,0,0,0,1],
            [0,1,1,1,0,1,1,1,0],
        ])),
        (5, np.array([ # 33-23-2 facet 
            [0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0],
        ])),
        (5, np.array([ # 33-23-2 facet
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 1, 0],
        ])),
        (7, np.array([ # 33-23-2 facet
            [0, 0, 1, 0, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 1, 0, 0],
        ])),
        mac.rac_game(2), # 2-bit rac
        mac.rac_game(3), # 3-bit rac (not tight)
        (8, np.array([  # 8-input dimensionality witness derived from 3-bit RAC
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        ])),  
    ]

    num_in_list = [
        (3, 2),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (4, 2),
        (8, 3),
        (8, 3),
    ]

    # fixed_setting_arb_ids = [
    #     [0,4,8],
    #     [0,4,8,12],
    #     [0,4,8,12],
    #     [0,4,8],
    #     [0,4,8,12],
    #     [0,4,8,12,16],
    #     [0,4,8,12,16,20],
    #     [0,4,8,12,16,20],
    # ]
    # fixed_setting_ry_ids = [
    #     [0,2,4],
    #     [0,2,4,6],
    #     [0,2,4,6],
    #     [0,2,4],
    #     [0,2,4,6],
    #     [0,2,4,6,8],
    #     [0,2,4,6,8,10],
    #     [0,2,4,6,8,10],
    # ]
    # fixed_settings = [
    #     [np.pi,0,0],[np.pi,0,0,0],
    #     [np.pi,0,0,0],
    #     [np.pi,0,0],
    #     [np.pi,0,0,0],
    #     # [0,0,np.pi,0],
    #     [np.pi,0,0,0,0],
    #     [np.pi,0,0,0,0,0],
    #     [np.pi,0,0,0,0,np.pi]
    # ]

    for i in range(0,len(inequalities)):
        inequality = inequalities[i]
        num_in = inequality[1].shape[1]

        print("i = ", i)
        inequality_tag = "I_" + str(i) + "_"



        n_workers = 3
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

        num_in_a = num_in_list[i][0]
        num_in_b = num_in_list[i][1]


        """
        Quantum Signaling
        """
        client.restart()

        time_start = time.time()

        qc_opt_fn = optimize_inequality(
            [
                qc_tx_nodes(num_in_a),
                qc_rx_nodes(num_in_b),
            ],
            postmap,
            inequality,
            # fixed_setting_ids=fixed_setting_ry_ids[i],
            # fixed_settings=fixed_settings[i],
            num_steps=150,
            step_size=0.75,
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

        scenario = "qc_arb_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        Entanglement-Assisted Classical Signaling
        """
        client.restart()
        
        time_start = time.time()

        eacc_opt_fn = optimize_inequality(
            [
                ghz_prep_node,
                eacc_tx_nodes(num_in_a, ansatz_fn=eacc_arb_tx, num_settings=4),
                eacc_rx_nodes(num_in_b, ansatz_fn=eacc_arb_rx, num_settings=30),
                twoqubit_measure_node,
            ],
            parity_postmap,
            inequality,
            # fixed_setting_ids=fixed_setting_ry_ids[i],
            # fixed_settings=fixed_settings[i],
            num_steps=150,
            step_size=0.5,
            sample_width=1,
            verbose=True,
        )

        eacc_opt_jobs = client.map(eacc_opt_fn, range(n_workers))
        eacc_opt_dicts = client.gather(eacc_opt_jobs)

        max_opt_dict = eacc_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eacc_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eacc_opt_dicts[j]["scores"])
                max_opt_dict = eacc_opt_dicts[j]

        scenario = "eacc_arb_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        Entanglement-Assisted Quantum Signaling
        """
        client.restart()
        
        time_start = time.time()

        eaqc_opt_fn = optimize_inequality(
            [
                ghz_prep_node,
                eaqc_tx_nodes(num_in_a),
                eaqc_rx_nodes(num_in_b),
            ],
            parity_postmap,
            inequality,
            num_steps=150,
            step_size=0.2,
            sample_width=1,
            verbose=True,
        )

        eaqc_opt_jobs = client.map(eaqc_opt_fn, range(n_workers))
        eaqc_opt_dicts = client.gather(eaqc_opt_jobs)

        max_opt_dict = eaqc_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eaqc_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eaqc_opt_dicts[j]["scores"])
                max_opt_dict = eaqc_opt_dicts[j]

        scenario = "eaqc_arb_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        # """
        # Two Entanglement-Assisted Quantum Signaling
        # """
        # client.restart()
        
        # time_start = time.time()

        # eaqc_opt_fn = optimize_inequality(
        #     [
        #         twoghz_prep_nodes,
        #         eaqc2_tx_nodes(num_in_a),
        #         eaqc2_rx_nodes(num_in_b),
        #     ],
        #     parity_postmap3,
        #     inequality,
        #     num_steps=150,
        #     step_size=0.2,
        #     sample_width=1,
        #     verbose=True,
        # )

        # eaqc_opt_jobs = client.map(eaqc_opt_fn, range(n_workers))
        # eaqc_opt_dicts = client.gather(eaqc_opt_jobs)

        # max_opt_dict = eaqc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(eaqc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(eaqc_opt_dicts[j]["scores"])
        #         max_opt_dict = eaqc_opt_dicts[j]

        # scenario = "eaqc2_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # Dense Coding GHZ RYRZ Arb Signaling Dimension
        # """
        # client.restart()
        
        # time_start = time.time()

        # eaqc_opt_fn = optimize_inequality(
        #     [
        #         ghz_prep_node,
        #         dense_encoder_nodes(num_in, ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        #         dense_decoder_arb_node,
        #     ],
        #     postmap,
        #     inequality,
        #     num_steps=150,
        #     step_size=0.3,
        #     sample_width=1,
        #     verbose=False
        # )

        # eaqc_opt_jobs = client.map(eaqc_opt_fn, range(n_workers))
        # eaqc_opt_dicts = client.gather(eaqc_opt_jobs)

        # max_opt_dict = eaqc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(eaqc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(eaqc_opt_dicts[j]["scores"])
        #         max_opt_dict = eaqc_opt_dicts[j]

        # scenario = "eaqc_ghz_ryrz_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # GHZ Dense Coding Arb Arb Signaling Dimension
        # """
        # client.restart()
        
        # time_start = time.time()

        # eaqc_opt_fn = optimize_inequality(
        #     [
        #         ghz_prep_node,
        #         dense_encoder_nodes(num_in, ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        #         dense_decoder_arb_node,
        #     ],
        #     postmap,
        #     inequality,
        #     num_steps=150,
        #     step_size=0.3,
        #     sample_width=1,
        #     verbose=False
        # )

        # eaqc_opt_jobs = client.map(eaqc_opt_fn, range(n_workers))
        # eaqc_opt_dicts = client.gather(eaqc_opt_jobs)

        # max_opt_dict = eaqc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(eaqc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(eaqc_opt_dicts[j]["scores"])
        #         max_opt_dict = eaqc_opt_dicts[j]

        # scenario = "eaqc_ghz_arb_arb_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # GHZACC Signaling Dimension
        # """
        # client.restart()

        # time_start = time.time()

        # ghzacc_opt_fn = optimize_inequality(
        #     [
        #         ghz_prep_node,
        #         ea_sender_nodes(num_in), #ansatz_fn=ea_ry_encoder, num_settings=2),
        #         ea_arb_decoder_node
        #     ],
        #     postmap,
        #     inequality,
        #     fixed_setting_ids=fixed_setting_ry_ids[i],
        #     fixed_settings=fixed_settings[i],
        #     num_steps=150,
        #     step_size=0.8,
        #     sample_width=1,
        #     verbose=False
        # )

        # ghzacc_opt_jobs = client.map(ghzacc_opt_fn, range(n_workers))
        # ghzacc_opt_dicts = client.gather(ghzacc_opt_jobs)

        # max_opt_dict = ghzacc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(ghzacc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(ghzacc_opt_dicts[j]["scores"])
        #         max_opt_dict = ghzacc_opt_dicts[j]

        # scenario = "ghzacc_ry_encoder_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

       