import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo

import context
import src

if __name__=="__main__":


    data_dir = "data/bipartite_broadcast_quantum_violations/"


    qubit_projector_postmap = np.array([[1,0],[0,1],[0,0]])
    qubit_povm_postmap = np.array([[1,0,1,0],[0,1,0,0],[0,0,0,1]])
    # qubit_povm_postmap = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1]])


    projector_postmap = np.kron(qubit_projector_postmap, qubit_projector_postmap)
    povm_postmap = np.kron(qubit_povm_postmap, qubit_povm_postmap)


    broadcast_proj_prep_nodes = [
        qnetvo.PrepareNode(
            num_in=3,
            wires=[0,1],
            ansatz_fn=qml.ArbitraryStatePreparation,
            num_settings=6,
        ),
    ]
    broadcast_ea_prep_node = [
        qnetvo.PrepareNode(
            wires=[0,1],
            ansatz_fn=qnetvo.ghz_state,
        )
    ]
    broadcast_proj_meas_nodes = [
        qnetvo.MeasureNode(
            num_out=3,
            wires=[0],
            ansatz_fn=qml.ArbitraryUnitary,
            num_settings=3,
        ),
        qnetvo.MeasureNode(
            num_out=3,
            wires=[1],
            ansatz_fn=qml.ArbitraryUnitary,
            num_settings=3,
        ),
    ]

    def classical_ea_prep_circ(settings, wires):
        qnetvo.ghz_state(settings, wires=[wires[0], wires[2]])
        qnetvo.ghz_state(settings, wires=[wires[1], wires[3]])

    broadcast_ea_prep_nodes = [
        qnetvo.PrepareNode(
            wires=[0,1,2,3],
            ansatz_fn=classical_ea_prep_circ,
        )
    ]
    broadcast_ea_proj_meas_nodes = [
        qnetvo.MeasureNode(
            num_in=2,
            num_out=3,
            wires=[0,1],
            ansatz_fn=qml.ArbitraryUnitary,
            num_settings=15,
        ),
        qnetvo.MeasureNode(
            num_in=2,
            num_out=3,
            wires=[2,3],
            ansatz_fn=qml.ArbitraryUnitary,
            num_settings=15,
        ),
    ]

    def povm_circ(settings, wires):
        qml.ArbitraryStatePreparation(settings, [wires[0], wires[2]])

    def ea_povm_circ(settings, wires):
        qml.ArbitraryStatePreparation(settings, [wires[0], wires[2]])
        qml.Hadamard(wires=wires[1])
        qml.CNOT(wires=[wires[1],wires[3]])






    broadcast_povm_prep_nodes = [
        qnetvo.PrepareNode(
            num_in=3,
            wires=[0,1,2,3],
            ansatz_fn=povm_circ,
            num_settings=6,
        ),
    ]

    broadcast_ea_povm_prep_nodes = [
        qnetvo.PrepareNode(
            num_in=3,
            wires=[0,1,2,3],
            ansatz_fn=ea_povm_circ,
            num_settings=6,
        )
    ]
    broadcast_povm_meas_nodes = [
        qnetvo.MeasureNode(
            num_out=3,
            wires=[0,1],
            ansatz_fn=qml.ArbitraryUnitary,
            num_settings=15,
        ),
        qnetvo.MeasureNode(
            num_out=3,
            wires=[2,3],
            ansatz_fn=qml.ArbitraryUnitary,
            num_settings=15,
        ),
    ]


    inequalities = src.bipartite_broadcast_bounds()
    
    for i in range(len(inequalities)):
        inequality = inequalities[i]

        print("i = ", i)
        inequality_tag = "I_" + str(i) + "_"

        client = Client(processes=True, n_workers=5, threads_per_worker=1)

        """
        Broadcast Projective Measurements
        """

        # time_start = time.time()

        # proj_broadcast_opt_fn = src.optimize_inequality(
        #     [
        #         broadcast_proj_prep_nodes,
        #         broadcast_proj_meas_nodes,
        #     ],
        #     projector_postmap,
        #     inequality,
        #     num_steps=150,
        #     step_size=0.15,
        #     sample_width=1,
        #     verbose=False,
        # )

        # proj_broadcast_opt_jobs = client.map(proj_broadcast_opt_fn, range(5))
        # proj_broadcast_opt_dicts = client.gather(proj_broadcast_opt_jobs)

        # proj_broadcast_max_opt_dict = proj_broadcast_opt_dicts[0]
        # proj_broadcast_max_score = max(proj_broadcast_max_opt_dict["scores"])
        # for j in range(1,5):
        #     if max(proj_broadcast_opt_dicts[j]["scores"]) > proj_broadcast_max_score:
        #         proj_broadcast_max_score = max(proj_broadcast_opt_dicts[j]["scores"])
        #         proj_broadcast_max_opt_dict = proj_broadcast_opt_dicts[j]

        # scenario = "proj_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     proj_broadcast_max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # Broadcast POVM measurements
        # """
        # time_start = time.time()

        # povm_broadcast_opt_fn = src.optimize_inequality(
        #     [
        #         broadcast_povm_prep_nodes,
        #         broadcast_povm_meas_nodes,
        #     ],
        #     povm_postmap,
        #     inequality,
        #     num_steps=150,
        #     step_size=0.15,
        #     sample_width=1,
        #     verbose=False,
        # )

        # povm_broadcast_opt_jobs = client.map(povm_broadcast_opt_fn, range(5))
        # povm_broadcast_opt_dicts = client.gather(povm_broadcast_opt_jobs)

        # povm_broadcast_max_opt_dict = povm_broadcast_opt_dicts[0]
        # povm_broadcast_max_score = max(povm_broadcast_max_opt_dict["scores"])
        # for j in range(1,5):
        #     if max(povm_broadcast_opt_dicts[j]["scores"]) > povm_broadcast_max_score:
        #         povm_broadcast_max_score = max(povm_broadcast_opt_dicts[j]["scores"])
        #         povm_broadcast_max_opt_dict = povm_broadcast_opt_dicts[j]

        # scenario = "povm_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     povm_broadcast_max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # Entanglement-Assisted POVM
        # """
        # time_start = time.time()

        # ea_povm_broadcast_opt_fn = src.optimize_inequality(
        #     [
        #         broadcast_ea_povm_prep_nodes,
        #         broadcast_povm_meas_nodes,
        #     ],
        #     povm_postmap,
        #     inequality,
        #     num_steps=150,
        #     step_size=0.15,
        #     sample_width=1,
        #     verbose=False,
        # )

        # ea_povm_broadcast_opt_jobs = client.map(ea_povm_broadcast_opt_fn, range(5))
        # ea_povm_broadcast_opt_dicts = client.gather(ea_povm_broadcast_opt_jobs)

        # ea_povm_broadcast_max_opt_dict = ea_povm_broadcast_opt_dicts[0]
        # ea_povm_broadcast_max_score = max(ea_povm_broadcast_max_opt_dict["scores"])
        # for j in range(1,5):
        #     if max(ea_povm_broadcast_opt_dicts[j]["scores"]) > ea_povm_broadcast_max_score:
        #         ea_povm_broadcast_max_score = max(ea_povm_broadcast_opt_dicts[j]["scores"])
        #         ea_povm_broadcast_max_opt_dict = ea_povm_broadcast_opt_dicts[j]

        # scenario = "ea_povm_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     ea_povm_broadcast_max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)


        """
        Entanglement-Assisted classical communication
        """
        time_start = time.time()

        lifted_inequality = (inequality[0], np.append(inequality[1], np.zeros((9,1)), axis=1))

        ea_proj_broadcast_opt_fn = src.optimize_inequality(
            [
                broadcast_ea_prep_nodes,
                broadcast_ea_proj_meas_nodes,
            ],
            povm_postmap,
            lifted_inequality,
            num_steps=150,
            step_size=0.15,
            sample_width=1,
            verbose=False,
        )

        ea_proj_broadcast_opt_jobs = client.map(ea_proj_broadcast_opt_fn, range(5))
        ea_proj_broadcast_opt_dicts = client.gather(ea_proj_broadcast_opt_jobs)

        ea_proj_broadcast_max_opt_dict = ea_proj_broadcast_opt_dicts[0]
        ea_proj_broadcast_max_score = max(ea_proj_broadcast_max_opt_dict["scores"])
        for j in range(1,5):
            if max(ea_proj_broadcast_opt_dicts[j]["scores"]) > ea_proj_broadcast_max_score:
                ea_proj_broadcast_max_score = max(ea_proj_broadcast_opt_dicts[j]["scores"])
                ea_proj_broadcast_max_opt_dict = ea_proj_broadcast_opt_dicts[j]

        scenario = "ea_proj_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            ea_proj_broadcast_max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)
