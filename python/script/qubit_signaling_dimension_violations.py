import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo

import context
import src

"""
The goal of this script is to identify quantum resource configurations in the bipartite point-to-point network
that can produce nonclassical behaviors. To achieve this goal, this script collects numerical optimization data for
maximizing nonclassicality against the computed set facet inequalities and simulation games for the considered 
scenario. Violations of these inequalities demonstrate a quantum advanttage over classical signaling.
"""

if __name__=="__main__":


    data_dir = "data/qubit_signaling_dimension/"

    postmap = np.eye(4)

    ghz_prep_node = [
        qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qnetvo.ghz_state),
    ]
    arb_prep_node = [
        qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]

    def ea_arb_encoder(settings, wires):
        # controlled use of entangled state
        qml.RY(settings[0], wires=[wires[2]])
        qml.ctrl(qml.SWAP, (wires[2]))(wires=wires[0:2])
        qml.ArbitraryUnitary(settings[1:4], wires=[wires[0]])
        
        b0 = qml.measure(wires[0])

        return [b0]
    
    def ea_ry_encoder(settings, wires):
        # controlled use of entangled state
        qml.RY(settings[0], wires=[wires[2]])
        qml.ctrl(qml.SWAP, (wires[2]))(wires=wires[0:2])
        qml.RY(settings[1], wires=[wires[0]])
        
        b0 = qml.measure(wires[0])

        return [b0]


    def ea_sender_nodes(num_in, ansatz_fn=ea_arb_encoder, num_settings=4):
        return [
            qnetvo.CCSenderNode(
                wires=[0,2,3],
                cc_wires_out=[0],
                num_in=num_in,
                ansatz_fn=ansatz_fn,
                num_settings=num_settings,
            ),
        ]
    
    def dense_encoder_nodes(num_in, ansatz_fn=qml.ArbitraryUnitary, num_settings=3):
        return [
            qnetvo.ProcessingNode(
                wires=[0],
                num_in=num_in,
                ansatz_fn=ansatz_fn,
                num_settings=num_settings,
            ),
        ]

    def ea_arb_decoder(settings, wires, cc_wires):

        qml.cond(
            cc_wires[0] == 0,
            qml.ArbitraryUnitary,
        )(settings[0:15], wires=wires[0:2])

        # qml.cond(cc_wires[0] == 1, qml.SWAP)(wires=wires[0:2])
        qml.cond(
            cc_wires[0] == 1,
            qml.ArbitraryUnitary,
        )(settings[15:30], wires=wires[0:2])


    ea_arb_decoder_node = [
        qnetvo.CCReceiverNode(
            wires=[1,4],
            cc_wires_in=[0],
            num_in=1,
            num_settings=30,
            ansatz_fn=ea_arb_decoder,
        )
    ]

    dense_decoder_arb_node = [
        qnetvo.MeasureNode(
            wires=[0,1],
            num_in=1,
            num_out=4,
            num_settings=15,
            ansatz_fn=qml.ArbitraryUnitary,
        ),
    ]

    # signaling dimension inequalities for qubit
    inequalities = src.qubit_signaling_dimension_bounds()

    fixed_setting_arb_ids = [
        [0,4,8],
        [0,4,8,12],
        [0,4,8,12],
        [0,4,8],
        [0,4,8,12],
        [0,4,8,12,16],
        [0,4,8,12,16,20],
        [0,4,8,12,16,20],
    ]
    fixed_setting_ry_ids = [
        [0,2,4],
        [0,2,4,6],
        [0,2,4,6],
        [0,2,4],
        [0,2,4,6],
        [0,2,4,6,8],
        [0,2,4,6,8,10],
        [0,2,4,6,8,10],
    ]
    fixed_settings = [
        [np.pi,0,0],[np.pi,0,0,0],
        [np.pi,0,0,0],
        [np.pi,0,0],
        [np.pi,0,0,0],
        # [0,0,np.pi,0],
        [np.pi,0,0,0,0],
        [np.pi,0,0,0,0,0],
        [np.pi,0,0,0,0,np.pi]
    ]

    for i in range(3,7):
        inequality = inequalities[i]
        num_in = inequality[1].shape[1]

        print("i = ", i)
        inequality_tag = "I_" + str(i) + "_"



        n_workers = 3
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

        # """
        # EACC Signaling Dimension
        # """
        # client.restart()
        
        # time_start = time.time()

        # eacc_opt_fn = src.optimize_inequality(
        #     [
        #         arb_prep_node,
        #         ea_sender_nodes(num_in, ansatz_fn=ea_ry_encoder, num_settings=2),
        #         ea_arb_decoder_node
        #     ],
        #     postmap,
        #     inequality,
        #     fixed_setting_ids=fixed_setting_ry_ids[i],
        #     fixed_settings=fixed_settings[i],
        #     num_steps=150,
        #     step_size=0.75,
        #     sample_width=1,
        #     verbose=False
        # )

        # eacc_opt_jobs = client.map(eacc_opt_fn, range(n_workers))
        # eacc_opt_dicts = client.gather(eacc_opt_jobs)

        # max_opt_dict = eacc_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(eacc_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(eacc_opt_dicts[j]["scores"])
        #         max_opt_dict = eacc_opt_dicts[j]

        # scenario = "eacc_ry_encoder_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # Arbitrary Dense Coding Signaling Dimension
        # """
        # client.restart()
        
        # time_start = time.time()

        # eaqc_opt_fn = src.optimize_inequality(
        #     [
        #         arb_prep_node,
        #         dense_encoder_nodes(num_in),
        #         dense_decoder_arb_node,
        #     ],
        #     postmap,
        #     inequality,
        #     num_steps=150,
        #     step_size=0.2,
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

        # scenario = "eaqc_arb_"
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

        # eaqc_opt_fn = src.optimize_inequality(
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

        # eaqc_opt_fn = src.optimize_inequality(
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

        """
        GHZACC Signaling Dimension
        """
        client.restart()

        time_start = time.time()

        ghzacc_opt_fn = src.optimize_inequality(
            [
                ghz_prep_node,
                ea_sender_nodes(num_in), #ansatz_fn=ea_ry_encoder, num_settings=2),
                ea_arb_decoder_node
            ],
            postmap,
            inequality,
            fixed_setting_ids=fixed_setting_ry_ids[i],
            fixed_settings=fixed_settings[i],
            num_steps=150,
            step_size=0.8,
            sample_width=1,
            verbose=False
        )

        ghzacc_opt_jobs = client.map(ghzacc_opt_fn, range(n_workers))
        ghzacc_opt_dicts = client.gather(ghzacc_opt_jobs)

        max_opt_dict = ghzacc_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(ghzacc_opt_dicts[j]["scores"]) > max_score:
                max_score = max(ghzacc_opt_dicts[j]["scores"])
                max_opt_dict = ghzacc_opt_dicts[j]

        scenario = "ghzacc_ry_encoder_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

       