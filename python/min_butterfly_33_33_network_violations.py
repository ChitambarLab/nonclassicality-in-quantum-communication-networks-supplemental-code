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


    data_dir = "data/min_butterfly_33_33_network_violations/"

    # postmap3 = np.array([
    #     [1,0,0,0],[0,1,0,0],[0,0,1,1],
    # ])
    postmap3 = np.array([
        [1,0,0,1],[0,1,0,0],[0,0,1,0],
    ])
    postmap38 = np.array([
        [1,1,0,0,0,0,0,0],
        [0,0,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1],
    ])

    qbf_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3])
    ]
    qbf_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0,2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        qnetvo.PrepareNode(num_in=3, wires=[1,3], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    qbf_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    qbf_layers = [
        qbf_wire_set_nodes,
        qbf_prep_nodes,
        qbf_meas_nodes,
    ]


    eatx_qbf_wires_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5,6])
    ]
    
    eatx_qbf_source_nodes = [
        qnetvo.PrepareNode(wires=[5,6], ansatz_fn=qnetvo.ghz_state)
    ]
    eatx_qbf_prep_nodes = [
        qnetvo.ProcessingNode(num_in=3, wires=[0,2,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        qnetvo.ProcessingNode(num_in=3, wires=[1,3,6], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]
    eatx_qbf_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.MeasureNode(num_out=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    eatx_qbf_layers = [
        eatx_qbf_wires_set_nodes,
        eatx_qbf_source_nodes,
        eatx_qbf_prep_nodes,
        eatx_qbf_meas_nodes,
    ]

    earx_qbf_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,5,6])
    ]
    earx_qbf_source_nodes = [
        qnetvo.PrepareNode(wires=[5,6], ansatz_fn=qnetvo.ghz_state),
    ]
    earx_qbf_tx_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0,2], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
        qnetvo.PrepareNode(num_in=3, wires=[1,3], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=6),
    ]
    earx_qbf_rx_nodes = [
        qnetvo.ProcessingNode(wires=[0,1,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
        qnetvo.ProcessingNode(wires=[2,3,6], ansatz_fn=qml.ArbitraryUnitary, num_settings=63),
    ]

    earx_qbf_meas_nodes = [
        qnetvo.MeasureNode(num_out=3, wires=[0,1]),
        qnetvo.MeasureNode(num_out=3, wires=[2,3]),
    ]

    earx_qbf_layers = [
        earx_qbf_wire_set_nodes,
        earx_qbf_tx_nodes,
        earx_qbf_rx_nodes,
        earx_qbf_meas_nodes,
    ]

    min_butterfly_game_inequalities, min_butterfly_facet_inequalities, game_names = mac.min_butterfly_33_33_network_bounds()

    for i in range(0,8):
        butterfly_game_inequality = min_butterfly_game_inequalities[i]
        butterfly_facet_inequality = min_butterfly_facet_inequalities[i]

        print("name = ", game_names[i])
        inequality_tag = "I_" + game_names[i] + "_"

        n_workers = 2
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        """
        quantum butterfly game
        """
        client.restart()

        time_start = time.time()

        postmap1 = postmap3
        postmap2 = postmap3

        qbf_game_opt_fn = optimize_inequality(
            qbf_layers,
            np.kron(postmap1,postmap2),
            butterfly_game_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        qbf_game_opt_jobs = client.map(qbf_game_opt_fn, range(n_workers))
        qbf_game_opt_dicts = client.gather(qbf_game_opt_jobs)

        max_opt_dict = qbf_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qbf_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qbf_game_opt_dicts[j]["scores"])
                max_opt_dict = qbf_game_opt_dicts[j]

        max_opt_dict["postmap1"] = postmap1.tolist()
        max_opt_dict["postmap2"] = postmap2.tolist()

        scenario = "qbf_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        quantum butterfly facet
        """
        client.restart()

        time_start = time.time()

        postmap1 = postmap3
        postmap2 = postmap3

        qbf_facet_opt_fn = optimize_inequality(
            qbf_layers,
            np.kron(postmap1,postmap2),
            butterfly_facet_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        qbf_facet_opt_jobs = client.map(qbf_facet_opt_fn, range(n_workers))
        qbf_facet_opt_dicts = client.gather(qbf_facet_opt_jobs)

        max_opt_dict = qbf_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(qbf_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(qbf_facet_opt_dicts[j]["scores"])
                max_opt_dict = qbf_facet_opt_dicts[j]

        max_opt_dict["postmap1"] = postmap1.tolist()
        max_opt_dict["postmap2"] = postmap2.tolist()

        scenario = "qbf_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)



        """
        eatx_quantum butterfly game
        """
        client.restart()

        time_start = time.time()

        postmap1 = postmap3
        postmap2 = postmap3

        eatx_qbf_game_opt_fn = optimize_inequality(
            eatx_qbf_layers,
            np.kron(postmap1,postmap2),
            butterfly_game_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        eatx_qbf_game_opt_jobs = client.map(eatx_qbf_game_opt_fn, range(n_workers))
        eatx_qbf_game_opt_dicts = client.gather(eatx_qbf_game_opt_jobs)

        max_opt_dict = eatx_qbf_game_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qbf_game_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qbf_game_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qbf_game_opt_dicts[j]
        
        max_opt_dict["postmap1"] = postmap1.tolist()
        max_opt_dict["postmap2"] = postmap2.tolist()

        scenario = "eatx_qbf_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        eatx quantum butterfly facet
        """
        client.restart()

        time_start = time.time()

        postmap1 = postmap3
        postmap2 = postmap3

        eatx_qbf_facet_opt_fn = optimize_inequality(
            eatx_qbf_layers,
            np.kron(postmap1,postmap2),
            butterfly_facet_inequality,
            num_steps=150,
            step_size=0.1,
            sample_width=1,
            verbose=True
        )

        eatx_qbf_facet_opt_jobs = client.map(eatx_qbf_facet_opt_fn, range(n_workers))
        eatx_qbf_facet_opt_dicts = client.gather(eatx_qbf_facet_opt_jobs)

        max_opt_dict = eatx_qbf_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qbf_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qbf_facet_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qbf_facet_opt_dicts[j]

        max_opt_dict["postmap1"] = postmap1.tolist()
        max_opt_dict["postmap2"] = postmap2.tolist()

        scenario = "eatx_qbf_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        # """
        # earx_quantum butterfly game
        # """
        # client.restart()

        # time_start = time.time()

        # postmap1 = postmap3
        # postmap2 = postmap3

        # earx_qbf_game_opt_fn = optimize_inequality(
        #     earx_qbf_layers,
        #     np.kron(postmap1,postmap2),
        #     butterfly_game_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # earx_qbf_game_opt_jobs = client.map(earx_qbf_game_opt_fn, range(n_workers))
        # earx_qbf_game_opt_dicts = client.gather(earx_qbf_game_opt_jobs)

        # max_opt_dict = earx_qbf_game_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(earx_qbf_game_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(earx_qbf_game_opt_dicts[j]["scores"])
        #         max_opt_dict = earx_qbf_game_opt_dicts[j]

        # max_opt_dict["postmap1"] = postmap1.tolist()
        # max_opt_dict["postmap2"] = postmap2.tolist()

        # scenario = "earx_qbf_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # earx quantum butterfly facet
        # """
        # client.restart()

        # time_start = time.time()

        # postmap1 = postmap3
        # postmap2 = postmap3

        # earx_qbf_facet_opt_fn = optimize_inequality(
        #     earx_qbf_layers,
        #     np.kron(postmap1,postmap2),
        #     butterfly_facet_inequality,
        #     num_steps=150,
        #     step_size=0.1,
        #     sample_width=1,
        #     verbose=True
        # )

        # earx_qbf_facet_opt_jobs = client.map(earx_qbf_facet_opt_fn, range(n_workers))
        # earx_qbf_facet_opt_dicts = client.gather(earx_qbf_facet_opt_jobs)

        # max_opt_dict = earx_qbf_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(earx_qbf_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(earx_qbf_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = earx_qbf_facet_opt_dicts[j]

        # max_opt_dict["postmap1"] = postmap1.tolist()
        # max_opt_dict["postmap2"] = postmap2.tolist()

        # scenario = "earx_qbf_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)