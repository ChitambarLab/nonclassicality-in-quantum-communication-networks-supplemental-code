import multiple_access_channels as mac
import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo


def adam_gradient_descent(
    cost,
    init_settings,
    num_steps=150,
    step_size=0.1,
    sample_width=25,
    grad_fn=None,
    verbose=True,
    interface="autograd",
):
    """
    adapted from qnetvo
    """

    if interface == "autograd":
        # opt = qml.GradientDescentOptimizer(stepsize=step_size)
        opt = qml.AdamOptimizer(stepsize=step_size)
    elif interface == "tf":
        from .lazy_tensorflow_import import tensorflow as tf

        opt = tf.keras.optimizers.SGD(learning_rate=step_size)
    else:
        raise ValueError('Interface "' + interface + '" is not supported.')

    settings = init_settings
    scores = []
    samples = []
    step_times = []
    settings_history = [init_settings]

    start_datetime = datetime.utcnow()
    elapsed = 0

    # performing gradient descent
    for i in range(num_steps):
        if i % sample_width == 0:
            score = -(cost(*settings))
            scores.append(score)
            samples.append(i)

            if verbose:
                print("iteration : ", i, ", score : ", score)

        start = time.time()
        if interface == "autograd":
            settings = opt.step(cost, *settings, grad_fn=grad_fn)
            if not (isinstance(settings, list)):
                settings = [settings]
        elif interface == "tf":
            # opt.minimize updates settings in place
            tf_cost = lambda: cost(*settings)
            opt.minimize(tf_cost, settings)

        elapsed = time.time() - start

        if i % sample_width == 0:
            step_times.append(elapsed)

            if verbose:
                print("elapsed time : ", elapsed)

        settings_history.append(settings)

    opt_score = -(cost(*settings))
    step_times.append(elapsed)

    scores.append(opt_score)
    samples.append(num_steps)

    return {
        "datetime": start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "opt_score": opt_score,
        "opt_settings": settings,
        "scores": scores,
        "samples": samples,
        "settings_history": settings_history,
        "step_times": step_times,
        "step_size": step_size,
    }

def _gradient_descent_wrapper(*opt_args, **opt_kwargs):
    """Wraps ``qnetvo.gradient_descent`` in a try-except block to gracefully
    handle errors during computation.
    This function is called with the same parameters as ``qnetvo.gradient_descent``.
    Optimization errors will result in an empty optimization dictionary.
    """
    try:
        # opt_dict = qnetvo.gradient_descent(*opt_args, **opt_kwargs)
        opt_dict = adam_gradient_descent(*opt_args, **opt_kwargs)
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


    data_dir = "data/interference2_33_33_network_violations/"

    postmap3 = np.array([
        [1,0,0,0],[0,1,0,0],[0,0,1,1],
    ])

    qint_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4]),
    ]
    qint_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        qnetvo.PrepareNode(num_in=3, wires=[1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
    ]
    qint_B_nodes = [
        qnetvo.ProcessingNode(wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    qint_C_nodes = [
        qnetvo.ProcessingNode(wires=[1,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    qint_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[3,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    eatx_qint_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6]),
    ]
    eatx_qint_source_nodes = [
        qnetvo.PrepareNode(wires=[1,4], ansatz_fn=qnetvo.ghz_state),
    ]
    eatx_qint_prep_nodes = [
        qnetvo.ProcessingNode(num_in=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.ProcessingNode(num_in=3, wires=[3,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    eatx_qint_B_nodes = [
        qnetvo.ProcessingNode(wires=[0,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    eatx_qint_C_nodes = [
        qnetvo.ProcessingNode(wires=[0,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    eatx_qint_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[5,6], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]



    # eatx_qint_source_nodes = [
    #     qnetvo.PrepareNode(wires=[0,1], ansatz_fn=qnetvo.ghz_state),
    # ]
    # eatx_qint_prep_nodes = [
    #     qnetvo.ProcessingNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    #     qnetvo.ProcessingNode(num_in=3, wires=[1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
    # ]


    # eatx_qint_source_nodes = [
    #     qnetvo.PrepareNode(wires=[0,3], ansatz_fn=qnetvo.ghz_state),
    # ]
    # eatx_qint_prep_nodes = [
    #     qnetvo.ProcessingNode(num_in=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15), 
    #     qnetvo.ProcessingNode(num_in=3, wires=[3,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=15), 
    # ]
    # eatx_qint_proc_nodes = [
    #     qnetvo.ProcessingNode(wires=[0,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    # ]
    # eatx_qint_meas_nodes = [
    #     qnetvo.MeasureNode(num_out=3, wires=[0,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    #     qnetvo.MeasureNode(num_out=3, wires=[3,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    # ]

    earx_qint_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4]),
    ]
    earx_qint_source_nodes = [
        qnetvo.PrepareNode(wires=[2,4], ansatz_fn=qnetvo.ghz_state),
    ]

    earx_qint_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        qnetvo.PrepareNode(num_in=3, wires=[1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
    ]
    earx_qint_B_nodes = [
        qnetvo.ProcessingNode(wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    earx_qint_C_nodes = [
        qnetvo.ProcessingNode(wires=[1,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    earx_qint_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[1,2], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[3,4], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]    


    interference_game_inequalities = [
        (7, np.array([ # multiplication with zero 
            [1, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])),
        (4 , np.array([ # multiplication game [1,2,3] no zero mult1
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])),
        (2, np.array([ # swap game
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])),
        (5, np.array([ # adder game
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])),
        (6, np.array([ # compare game
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])),
        (2, np.array([ # one receiver permutes output based on other receiver
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
        ])),
        (7, np.array([ # same difference game
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
        ])),
        (2, np.eye(9)), # communication value
    ]
    
    interference_facet_inequalities = [
        (12, np.array([ # mult 0 facet
            [1,  2,  2,  1,  0,  0,  0,  0,  1],
            [0,  0,  1,  0,  3,  0,  1,  2,  1],
            [0,  0,  0,  0,  0,  3,  1,  3,  0],
            [0,  1,  2,  0,  1,  2,  1,  2,  1],
            [0,  1,  2,  0,  1,  2,  1,  2,  1],
            [0,  1,  2,  0,  1,  2,  1,  2,  1],
            [0,  1,  2,  0,  1,  2,  1,  2,  1],
            [0,  1,  2,  0,  1,  2,  1,  2,  1],
            [1,  1,  2,  0,  2,  2,  1,  2,  0],
        ])),
        (13, np.array([ # mult 1 facet
            [4,  0,  1,  1,  0,  2,  0,  0,  1],
            [1,  3,  0,  3,  0,  0,  0,  0,  1],
            [0,  0,  3,  0,  2,  1,  1,  0,  0],
            [2,  1,  1,  2,  2,  1,  0,  0,  1],
            [2,  1,  2,  2,  1,  2,  0,  0,  1],
            [2,  1,  1,  2,  0,  3,  0,  1,  0],
            [2,  1,  2,  2,  1,  2,  0,  0,  1],
            [2,  1,  2,  2,  1,  2,  0,  0,  1],
            [3,  2,  2,  2,  1,  2,  0,  0,  0],
        ])),
        (9, np.array([ # swap facet
            [3,  0,  0,  0,  0,  1,  0,  0,  1],
            [0,  2,  0,  2,  1,  0,  0,  0,  1],
            [0,  1,  0,  0,  1,  2,  1,  0,  0],
            [0,  3,  0,  0,  0,  1,  0,  0,  1],
            [1,  1,  0,  1,  2,  0,  0,  0,  1],
            [1,  1,  0,  1,  0,  2,  0,  1,  0],
            [0,  1,  2,  0,  1,  0,  0,  1,  0],
            [1,  1,  0,  1,  0,  2,  0,  1,  0],
            [2,  2,  1,  1,  1,  1,  0,  0,  0],
        ])),
        (13, np.array([ # adder game
            [4,  0,  1,  1,  0,  2,  0,  0,  1],
            [1,  3,  0,  3,  0,  0,  0,  0,  1],
            [0,  0,  3,  0,  2,  1,  1,  0,  0],
            [2,  1,  1,  2,  0,  3,  0,  1,  0],
            [2,  1,  2,  2,  1,  2,  0,  0,  1],
            [2,  1,  2,  2,  1,  2,  0,  0,  1],
            [2,  1,  2,  2,  1,  2,  0,  0,  1],
            [2,  1,  2,  2,  1,  2,  0,  0,  1],
            [3,  2,  2,  2,  1,  2,  0,  0,  0],
        ])),
        (12, np.array([ # compare facet
            [3,  0,  0,  0,  2,  0,  0,  0,  1],
            [1,  0,  3,  2,  1,  2,  0,  0,  1],
            [1,  0,  3,  2,  1,  2,  0,  0,  1],
            [1,  0,  3,  2,  1,  2,  0,  0,  1],
            [1,  0,  3,  2,  1,  2,  0,  0,  1],
            [0,  2,  3,  1,  0,  3,  0,  0,  0],
            [1,  0,  3,  2,  1,  2,  0,  0,  1],
            [0,  0,  3,  3,  0,  2,  0,  1,  0],
            [2,  1,  3,  2,  1,  2,  0,  0,  0],
        ])),
        (9, np.array([ # conditioned permutation facett
            [3,  0,  0,  0,  0,  1,  0,  0,  1],
            [0,  3,  0,  0,  0,  1,  0,  0,  1],
            [0,  1,  2,  0,  1,  0,  0,  1,  0],
            [1,  1,  0,  1,  2,  0,  0,  0,  1],
            [1,  1,  0,  1,  0,  2,  0,  1,  0],
            [0,  2,  0,  2,  1,  0,  0,  0,  1],
            [1,  1,  1,  1,  1,  1,  0,  0,  1],
            [0,  1,  0,  0,  1,  2,  1,  0,  0],
            [2,  2,  1,  1,  1,  1,  0,  0,  0],
        ])),
        (11, np.array([ # same difference facet
            [2,  0,  0,  0,  2,  0,  0,  0,  1],
            [0,  1,  3,  1,  1,  2,  0,  0,  1],
            [0,  1,  3,  1,  1,  2,  0,  0,  1],
            [0,  1,  3,  1,  1,  2,  0,  0,  1],
            [0,  1,  3,  2,  0,  2,  0,  1,  0],
            [0,  1,  3,  1,  1,  2,  0,  0,  1],
            [0,  1,  3,  1,  1,  2,  0,  0,  1],
            [0,  1,  3,  1,  1,  2,  0,  0,  1],
            [1,  2,  3,  1,  1,  2,  0,  0,  0],
        ])),
        (9, np.array([ # cv facet
            [3,  0,  0,  0,  0,  1,  0,  0,  1],
            [0,  3,  0,  0,  0,  1,  0,  0,  1],
            [0,  1,  2,  0,  1,  0,  0,  1,  0],
            [0,  2,  0,  2,  1,  0,  0,  0,  1],
            [1,  1,  0,  1,  2,  0,  0,  0,  1],
            [1,  1,  0,  1,  0,  2,  0,  1,  0],
            [0,  1,  0,  0,  1,  2,  1,  0,  0],
            [1,  1,  0,  1,  0,  2,  0,  1,  0],
            [2,  2,  1,  1,  1,  1,  0,  0,  0],
        ])),
    ]

    game_names = ["mult0", "mult1", "swap", "adder", "compare", "perm", "diff", "cv"]
    

    for i in range(0,8):
        interference_game_inequality = interference_game_inequalities[i]
        interference_facet_inequality = interference_facet_inequalities[i]

        print("name = ", game_names[i])
        inequality_tag = "I_" + game_names[i] + "_"

        n_workers = 1
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        """
        quantum interference game
        """
        client.restart()

        time_start = time.time()

        qint_game_opt_fn = optimize_inequality(
            [
                qint_wire_set_nodes,
                qint_prep_nodes,
                qint_B_nodes,
                qint_C_nodes,
                qint_meas_nodes,
            ],
            np.kron(postmap3,postmap3),
            interference_game_inequality,
            num_steps=150,
            step_size=0.08,
            sample_width=1,
            verbose=True
        )

        qint_game_opt_jobs = client.map(qint_game_opt_fn, range(n_workers))
        qint_game_opt_dicts = client.gather(qint_game_opt_jobs)

        max_opt_dict = qint_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qint_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qint_game_opt_dicts[j]["scores"])
                max_opt_dict = qint_game_opt_dicts[j]

        scenario = "qint_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        quantum interference facet
        """
        client.restart()

        time_start = time.time()

        qint_facet_opt_fn = optimize_inequality(
            [
                qint_wire_set_nodes,
                qint_prep_nodes,
                qint_B_nodes,
                qint_C_nodes,
                qint_meas_nodes,
            ],
            np.kron(postmap3,postmap3),
            interference_facet_inequality,
            num_steps=150,
            step_size=0.08,
            sample_width=1,
            verbose=True
        )

        qint_facet_opt_jobs = client.map(qint_facet_opt_fn, range(n_workers))
        qint_facet_opt_dicts = client.gather(qint_facet_opt_jobs)

        max_opt_dict = qint_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qint_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qint_facet_opt_dicts[j]["scores"])
                max_opt_dict = qint_facet_opt_dicts[j]

        scenario = "qint_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)
        
        """
        eatx quantum interference game
        """
        client.restart()

        time_start = time.time()

        eatx_qint_game_opt_fn = optimize_inequality(
            [
                eatx_qint_wire_set_nodes,
                eatx_qint_source_nodes,
                eatx_qint_prep_nodes,
                eatx_qint_B_nodes,
                eatx_qint_C_nodes,
                eatx_qint_meas_nodes,
            ],
            np.kron(postmap3,postmap3),
            interference_game_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        eatx_qint_game_opt_jobs = client.map(eatx_qint_game_opt_fn, range(n_workers))
        eatx_qint_game_opt_dicts = client.gather(eatx_qint_game_opt_jobs)

        max_opt_dict = eatx_qint_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qint_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qint_game_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qint_game_opt_dicts[j]

        scenario = "eatx_qint_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        quantum interference facet
        """
        client.restart()

        time_start = time.time()

        eatx_qint_facet_opt_fn = optimize_inequality(
            [
                eatx_qint_wire_set_nodes,
                eatx_qint_source_nodes,
                eatx_qint_prep_nodes,
                eatx_qint_B_nodes,
                eatx_qint_C_nodes,
                eatx_qint_meas_nodes,
            ],
            np.kron(postmap3,postmap3),
            interference_facet_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        eatx_qint_facet_opt_jobs = client.map(eatx_qint_facet_opt_fn, range(n_workers))
        eatx_qint_facet_opt_dicts = client.gather(eatx_qint_facet_opt_jobs)

        max_opt_dict = eatx_qint_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qint_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qint_facet_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qint_facet_opt_dicts[j]

        scenario = "eatx_qint_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        earx quantum interference game
        """
        client.restart()

        time_start = time.time()

        earx_qint_game_opt_fn = optimize_inequality(
            [
                earx_qint_wire_set_nodes,
                earx_qint_source_nodes,
                earx_qint_prep_nodes,
                earx_qint_B_nodes,
                earx_qint_C_nodes,
                earx_qint_meas_nodes,
            ],
            np.kron(postmap3,postmap3),
            interference_game_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        earx_qint_game_opt_jobs = client.map(earx_qint_game_opt_fn, range(n_workers))
        earx_qint_game_opt_dicts = client.gather(earx_qint_game_opt_jobs)

        max_opt_dict = earx_qint_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(earx_qint_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(earx_qint_game_opt_dicts[j]["scores"])
                max_opt_dict = earx_qint_game_opt_dicts[j]

        scenario = "earx_qint_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        earx quantum interference facet
        """
        client.restart()

        time_start = time.time()

        earx_qint_facet_opt_fn = optimize_inequality(
            [
                earx_qint_wire_set_nodes,
                earx_qint_source_nodes,
                earx_qint_prep_nodes,
                earx_qint_B_nodes,
                earx_qint_C_nodes,
                earx_qint_meas_nodes,
            ],
            np.kron(postmap3,postmap3),
            interference_facet_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        earx_qint_facet_opt_jobs = client.map(earx_qint_facet_opt_fn, range(n_workers))
        earx_qint_facet_opt_dicts = client.gather(earx_qint_facet_opt_jobs)

        max_opt_dict = earx_qint_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(earx_qint_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(earx_qint_facet_opt_dicts[j]["scores"])
                max_opt_dict = earx_qint_facet_opt_dicts[j]

        scenario = "earx_qint_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)
        