import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo

import context
import src

if __name__=="__main__":


    data_dir = "data/qubit_interference_network/"

    postmap3 = np.array([
        [1,0,0,0],[0,1,0,0],[0,0,1,1],
    ])
    postmap2 = np.array([
        [1,0],[0,1],
    ])


    def ghz_ansatz(settings, wires):
        # pass
        qnetvo.ghz_state(settings, wires=[wires[0],wires[2]])

    wire_set_prep_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3], ansatz_fn=ghz_ansatz),
    ]
    wire_set_prep_nodes2 = [
        qnetvo.PrepareNode(wires=[0,1,2], ansatz_fn=ghz_ansatz),
    ]
    qubit_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        qnetvo.PrepareNode(num_in=3, wires=[2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
    ]


    interference_node = [
        qnetvo.ProcessingNode(wires=[0,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15)
    ]

    measure_nodes = [
        qnetvo.MeasureNode(wires=[0, 1], num_out=3, ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(wires=[2, 3], num_out=3, ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    measure_nodes2 = [
        qnetvo.MeasureNode(wires=[0, 1], num_out=3, ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(wires=[2], num_out=2, ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    ]
    
    # signaling dimension inequalities for qubit
    inequalities = [
        (13, np.array([
                [3,  0,  0,  0,  2,  0,  1,  0,  1],
                [0,  3,  0,  2,  0,  0,  0,  1,  1],
                [0,  0,  4,  1,  1,  0,  1,  1,  0],
                [1,  1,  2,  2,  0,  0,  1,  0,  1],
                [1,  1,  2,  0,  2,  0,  0,  1,  1],
                [1,  2,  2,  0,  0,  2,  1,  1,  0],
                [0,  1,  3,  0,  1,  1,  2,  0,  0],
                [1,  0,  3,  1,  0,  1,  0,  2,  0],
                [2,  2,  3,  1,  1,  1,  1,  1,  0],
            ])
        ),
    ]

    inequalities2 = [
        (14, np.array([
                [3,  0,  1,  0,  0,  1,  0,  1,  2],
                [0,  3,  0,  2,  1,  0,  0,  1,  2],
                [0,  0,  4,  0,  2,  0,  2,  0,  0],
                [2,  1,  2,  0,  1,  2,  0,  2,  0],
                [2,  2,  2,  1,  1,  0,  0,  0,  2],
                [2,  2,  3,  1,  1,  1,  1,  1,  1],
            ])
        ),
    ]
    

    for i in range(0,1):
        inequality = inequalities2[i]
        num_in = inequality[1].shape[1]

        print("i = ", i)
        inequality_tag = "I_" + str(i) + "_"



        n_workers = 1
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        """
        qubit interference
        """
        client.restart()

        time_start = time.time()

        qint_opt_fn = src.optimize_inequality(
            [
                wire_set_prep_nodes2,
                qubit_prep_nodes,
                interference_node,
                measure_nodes2,
            ],
            np.kron(postmap3,postmap2),
            inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        qint_opt_jobs = client.map(qint_opt_fn, range(n_workers))
        qint_opt_dicts = client.gather(qint_opt_jobs)

        max_opt_dict = qint_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qint_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qint_opt_dicts[j]["scores"])
                max_opt_dict = qint_opt_dicts[j]

        scenario = "qint_encoder_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

       