import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo

import context
import src

"""
The goal of this script is to identify quantum resource configurations in the bipartite prepare-and-measure scenario 
that can produce nonclassical behaviors with respect to the random access coding game. To achieve this goal,
this script collects numerical optimization data for maximizing nonclassicality against the computed set facet inequalities
and simulation games for the considered  prepare and measure scenario. Violations of these inequalities demonstrate a
quantum advanttage over classical signaling.
"""

if __name__=="__main__":


    data_dir = "data/qubit_random_access_coding/"

    postmap = np.eye(2)

    for n in range(2,7):
        rac_inequality = src.rac_game(n)
        print(rac_inequality[1])


        num_in = rac_inequality[1].shape[1]

        print("n = ", n)
        print("bounds = ", rac_inequality[0])
        inequality_tag = "I_n_" + str(n) + "_"

        n_workers = 3
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

        """
        qubit siganling
        """
        qubit_prep_nodes = [
            qnetvo.PrepareNode(num_in = 2**n, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        ]
        qubit_meas_nodes = [
            qnetvo.MeasureNode(num_in=n, num_out=2, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
        ]

        client.restart()

        time_start = time.time()

        rac_n_opt_fn = src.optimize_inequality(
            [
                qubit_prep_nodes,
                qubit_meas_nodes,
            ],
            postmap,
            rac_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        rac_n_opt_jobs = client.map(rac_n_opt_fn, range(n_workers))
        rac_n_opt_dicts = client.gather(rac_n_opt_jobs)

        max_opt_dict = rac_n_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(rac_n_opt_dicts[j]["scores"]) > max_score:
                max_score = max(rac_n_opt_dicts[j]["scores"])
                max_opt_dict = rac_n_opt_dicts[j]

        scenario = "rac_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

       