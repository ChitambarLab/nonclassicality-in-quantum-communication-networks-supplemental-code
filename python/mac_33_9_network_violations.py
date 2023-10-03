import multiple_access_channels as mac
import pennylane as qml
from pennylane import numpy as np
from dask.distributed import Client
import time
from datetime import datetime

import qnetvo



# def adam_gradient_descent(
#     cost,
#     init_settings,
#     num_steps=150,
#     step_size=0.1,
#     sample_width=25,
#     grad_fn=None,
#     verbose=True,
#     interface="autograd",
# ):
#     """
#     adapted from qnetvo
#     """

#     if interface == "autograd":
#         # opt = qml.GradientDescentOptimizer(stepsize=step_size)
#         opt = qml.AdamOptimizer(stepsize=step_size)
#     elif interface == "tf":
#         from .lazy_tensorflow_import import tensorflow as tf

#         opt = tf.keras.optimizers.SGD(learning_rate=step_size)
#     else:
#         raise ValueError('Interface "' + interface + '" is not supported.')

#     settings = init_settings
#     scores = []
#     samples = []
#     step_times = []
#     settings_history = [init_settings]

#     start_datetime = datetime.utcnow()
#     elapsed = 0

#     # performing gradient descent
#     for i in range(num_steps):
#         if i % sample_width == 0:
#             score = -(cost(*settings))
#             scores.append(score)
#             samples.append(i)

#             if verbose:
#                 print("iteration : ", i, ", score : ", score)

#         start = time.time()
#         if interface == "autograd":
#             settings = opt.step(cost, *settings, grad_fn=grad_fn)
#             if not (isinstance(settings, list)):
#                 settings = [settings]
#         elif interface == "tf":
#             # opt.minimize updates settings in place
#             tf_cost = lambda: cost(*settings)
#             opt.minimize(tf_cost, settings)

#         elapsed = time.time() - start

#         if i % sample_width == 0:
#             step_times.append(elapsed)

#             if verbose:
#                 print("elapsed time : ", elapsed)

#         settings_history.append(settings)

#     opt_score = -(cost(*settings))
#     step_times.append(elapsed)

#     scores.append(opt_score)
#     samples.append(num_steps)

#     return {
#         "datetime": start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
#         "opt_score": opt_score,
#         "opt_settings": settings,
#         "scores": scores,
#         "samples": samples,
#         "settings_history": settings_history,
#         "step_times": step_times,
#         "step_size": step_size,
#     }



def _gradient_descent_wrapper(*opt_args, **opt_kwargs):
    """Wraps ``qnetvo.gradient_descent`` in a try-except block to gracefully
    handle errors during computation.
    This function is called with the same parameters as ``qnetvo.gradient_descent``.
    Optimization errors will result in an empty optimization dictionary.
    """
    try:
        opt_dict = qnetvo.gradient_descent(*opt_args, **opt_kwargs, optimizer="adam")
        # opt_dict = adam_gradient_descent(*opt_args, **opt_kwargs)
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


    data_dir = "data/mac_33_9_network_violations/"

    postmap9 = np.array([
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    ])
    postmap9m0 = np.array([
        [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    qmac_wire_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3]),
    ]
    qmac_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
        qnetvo.PrepareNode(num_in=3, wires=[1], ansatz_fn=qml.ArbitraryStatePreparation, num_settings=2),
    ]
    qmac_meas_nodes = [
        qnetvo.MeasureNode(num_out=9,wires=[0,1,2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=255),
    ]

    qmac_layers = [
        qmac_wire_set_nodes,
        qmac_prep_nodes,
        qmac_meas_nodes,
    ]
    
    eatx_wires_set_nodes = [
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5]),
    ]
    eatx_qmac_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2], ansatz_fn=qnetvo.ghz_state),
    ]

    eatx_qmac_prep_nodes = [
        qnetvo.ProcessingNode(num_in=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.ProcessingNode(num_in=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]

    eatx_qmac_meas_nodes = [
        qnetvo.MeasureNode(num_out=9,wires=[0,2,4,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=255),
    ]

    eatx_qmac_layers = [
        eatx_wires_set_nodes,
        eatx_qmac_source_nodes,
        eatx_qmac_prep_nodes,
        eatx_qmac_meas_nodes,
    ]

    ghza_qmac_wire_set_nodes =[
        qnetvo.PrepareNode(wires=[0,1,2,3,4,5]),
    ]
    ghza_qmac_source_nodes = [
        qnetvo.PrepareNode(wires=[0,2,4], ansatz_fn=qnetvo.ghz_state),
    ]
    ghza_qmac_prep_nodes = [
        qnetvo.PrepareNode(num_in=3, wires=[0,1], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
        qnetvo.PrepareNode(num_in=3, wires=[2,3], ansatz_fn=qml.ArbitraryUnitary, num_settings=15),
    ]
    ghza_qmac_meas_nodes = [
        qnetvo.MeasureNode(num_out=9, wires=[0,2,4,5], ansatz_fn=qml.ArbitraryUnitary, num_settings=255),
    ]

    ghza_qmac_layers = [
        ghza_qmac_wire_set_nodes,
        ghza_qmac_source_nodes,
        ghza_qmac_prep_nodes,
        ghza_qmac_meas_nodes,
    ]


    mac_game_inequalities, mac_facet_inequalities, game_names = mac.mac_33_22_9_network_bounds()
    # mac_game_inequalities = [
    #     (7, np.array([ # multiplication with zero 
    #         [1, 1, 1, 1, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     ])),
    #     (5 , np.array([ # multiplication game [1,2,3] no zero mult1
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     ])),
    #     (4, np.array([ # swap game
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0],
    #         [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     ])),
    #     (5, np.array([ # adder game
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     ])),
    #     (7, np.array([ # compare game
    #         [1, 0, 0, 0, 1, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 1, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 1, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     ])),
    #     (4, np.array([ # one receiver permutes output based on other receiver
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 1, 0],
    #     ])),
    #     (7, np.array([ # same difference game
    #         [1, 0, 0, 0, 1, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 0, 1, 0, 1, 0, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0, 1, 0, 0],
    #     ])),
    #     (4, np.eye(9)), # communication value
    # ]
    
    # mac_facet_inequalities = [
    #     (10, np.array([ # mult zero face
    #         [0,  2,  1,  2,  0,  0,  0,  0,  1],
    #         [0,  0,  0,  0,  3,  0,  0,  0,  2],
    #         [0,  0,  0,  0,  1,  2,  0,  2,  0],
    #         [0,  0,  0,  0,  3,  0,  0,  0,  2],
    #         [0,  0,  0,  0,  3,  0,  0,  0,  2],
    #         [0,  0,  0,  0,  3,  0,  0,  0,  2],
    #         [0,  0,  0,  0,  3,  0,  0,  0,  2],
    #         [0,  0,  0,  0,  3,  0,  0,  0,  2],
    #         [0,  1,  1,  1,  2,  1,  1,  1,  1],
    #     ])),
    #     (10, np.array([ # mult 1 facet
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [0,  2,  0,  2,  0,  0,  0,  0,  2],
    #         [0,  0,  2,  0,  2,  0,  2,  0,  0],
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [2,  0,  0,  0,  0,  2,  0,  2,  0],
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [1,  1,  1,  1,  1,  1,  1,  1,  1],
    #     ])),
    #     (10, np.array([ # swap facet
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [0,  2,  0,  2,  0,  0,  0,  0,  2],
    #         [0,  2,  0,  0,  0,  2,  2,  0,  0],
    #         [0,  2,  0,  2,  0,  0,  0,  0,  2],
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [2,  0,  0,  0,  0,  2,  0,  2,  0],
    #         [0,  0,  2,  2,  0,  0,  0,  2,  0],
    #         [2,  0,  0,  0,  0,  2,  0,  2,  0],
    #         [1,  1,  1,  1,  1,  1,  1,  1,  1],
    #     ])),
    #     (10, np.array([ # adder facet
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [0,  2,  0,  2,  0,  0,  0,  0,  2],
    #         [0,  0,  2,  0,  2,  0,  2,  0,  0],
    #         [2,  0,  0,  0,  0,  2,  0,  2,  0],
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [1,  1,  1,  1,  1,  1,  1,  1,  1],
    #     ])),
    #     (9, np.array([ # compare facet
    #         [2,  0,  0,  0,  2,  0,  0,  0,  1],
    #         [1,  1,  0,  1,  0,  1,  0,  1,  1],
    #         [1,  1,  0,  1,  0,  1,  0,  1,  1],
    #         [1,  1,  0,  1,  0,  1,  0,  1,  1],
    #         [1,  1,  0,  1,  0,  1,  0,  1,  1],
    #         [0,  2,  0,  0,  0,  2,  1,  0,  0],
    #         [1,  1,  0,  1,  0,  1,  0,  1,  1],
    #         [0,  0,  1,  2,  0,  0,  0,  2,  0],
    #         [1,  1,  1,  1,  1,  1,  1,  1,  0],
    #     ])),
    #     (10, np.array([ # conditioned permutation facet
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [0,  2,  0,  1,  0,  0,  1,  0,  2],
    #         [0,  0,  2,  1,  1,  0,  1,  1,  0],
    #         [1,  0,  0,  0,  2,  0,  1,  0,  2],
    #         [1,  1,  0,  0,  0,  2,  1,  1,  0],
    #         [0,  2,  0,  2,  0,  0,  0,  0,  2],
    #         [1,  0,  0,  0,  2,  0,  1,  0,  2],
    #         [0,  1,  1,  0,  1,  1,  2,  0,  1],
    #         [1,  1,  1,  1,  1,  1,  1,  1,  1],
    #     ])),
    #     (9, np.array([ # same difference facet
    #         [2,  0,  0,  0,  2,  0,  0,  0,  1],
    #         [1,  0,  1,  1,  0,  2,  0,  0,  1],
    #         [1,  0,  1,  1,  0,  2,  0,  0,  1],
    #         [1,  0,  1,  1,  0,  2,  0,  0,  1],
    #         [0,  0,  1,  2,  0,  1,  0,  1,  0],
    #         [1,  0,  1,  1,  0,  2,  0,  0,  1],
    #         [1,  0,  1,  1,  0,  2,  0,  0,  1],
    #         [1,  0,  1,  1,  0,  2,  0,  0,  1],
    #         [1,  1,  1,  1,  1,  2,  1,  0,  0],
    #     ])),
    #     (10, np.array([ # cv facet
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [0,  2,  0,  2,  0,  0,  0,  0,  2],
    #         [0,  0,  2,  2,  0,  0,  0,  2,  0],
    #         [0,  2,  0,  2,  0,  0,  0,  0,  2],
    #         [2,  0,  0,  0,  2,  0,  0,  0,  2],
    #         [2,  0,  0,  0,  0,  2,  0,  2,  0],
    #         [0,  2,  0,  0,  0,  2,  2,  0,  0],
    #         [2,  0,  0,  0,  0,  2,  0,  2,  0],
    #         [1,  1,  1,  1,  1,  1,  1,  1,  1],
    #     ])),
    # ]

    # game_names = ["mult0", "mult1", "swap", "adder", "compare", "perm", "diff", "cv"]
    

    for i in range(0,8):
        mac_game_inequality = mac_game_inequalities[i]
        mac_facet_inequality = mac_facet_inequalities[i]

        print("name = ", game_names[i])
        inequality_tag = "I_" + game_names[i] + "_"

        n_workers = 2
        client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)


        # """
        # quantum mac game
        # """
        # client.restart()

        # time_start = time.time()

        # qmac_game_opt_fn = optimize_inequality(
        #     qmac_layers,
        #     # postmap9,
        #     postmap9m0,
        #     mac_game_inequality,
        #     num_steps=150,
        #     step_size=0.06,
        #     sample_width=1,
        #     verbose=True
        # )

        # qmac_game_opt_jobs = client.map(qmac_game_opt_fn, range(n_workers))
        # qmac_game_opt_dicts = client.gather(qmac_game_opt_jobs)

        # max_opt_dict = qmac_game_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qmac_game_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qmac_game_opt_dicts[j]["scores"])
        #         max_opt_dict = qmac_game_opt_dicts[j]

        # scenario = "qmac_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # quantum mac facet
        # """
        # client.restart()

        # time_start = time.time()

        # qmac_facet_opt_fn = optimize_inequality(
        #     qmac_layers,
        #     postmap9,
        #     mac_facet_inequality,
        #     num_steps=150,
        #     step_size=0.06,
        #     sample_width=1,
        #     verbose=True
        # )

        # qmac_facet_opt_jobs = client.map(qmac_facet_opt_fn, range(n_workers))
        # qmac_facet_opt_dicts = client.gather(qmac_facet_opt_jobs)

        # max_opt_dict = qmac_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(qmac_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(qmac_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = qmac_facet_opt_dicts[j]

        # scenario = "qmac_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        """
        eatx quantum mac game
        """
        client.restart()

        time_start = time.time()

        eatx_qmac_facet_opt_fn = optimize_inequality(
            eatx_qmac_layers,
            # postmap9,
            postmap9m0,
            mac_game_inequality,
            num_steps=150,
            step_size=0.06,
            sample_width=1,
            verbose=True
        )

        eatx_qmac_facet_opt_jobs = client.map(eatx_qmac_facet_opt_fn, range(n_workers))
        eatx_qmac_facet_opt_dicts = client.gather(eatx_qmac_facet_opt_jobs)

        max_opt_dict = eatx_qmac_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qmac_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qmac_facet_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qmac_facet_opt_dicts[j]

        scenario = "eatx_qmac_game_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)

        """
        eatx quantum mac facet
        """
        client.restart()

        time_start = time.time()

        eatx_qmac_facet_opt_fn = optimize_inequality(
            eatx_qmac_layers,
            postmap9,
            mac_facet_inequality,
            num_steps=150,
            step_size=0.06,
            sample_width=1,
            verbose=True
        )

        eatx_qmac_facet_opt_jobs = client.map(eatx_qmac_facet_opt_fn, range(n_workers))
        eatx_qmac_facet_opt_dicts = client.gather(eatx_qmac_facet_opt_jobs)

        max_opt_dict = eatx_qmac_facet_opt_dicts[0]
        max_score = max(max_opt_dict["scores"])
        for j in range(1,n_workers):
            if max(eatx_qmac_facet_opt_dicts[j]["scores"]) > max_score:
                max_score = max(eatx_qmac_facet_opt_dicts[j]["scores"])
                max_opt_dict = eatx_qmac_facet_opt_dicts[j]

        scenario = "eatx_qmac_facet_"
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        qnetvo.write_optimization_json(
            max_opt_dict,
            data_dir + scenario + inequality_tag + datetime_ext,
        )

        print("iteration time  : ", time.time() - time_start)


        # """
        # ghz quantum mac game
        # """
        # client.restart()

        # time_start = time.time()

        # ghza_qmac_facet_opt_fn = optimize_inequality(
        #     ghza_qmac_layers,
        #     postmap9,
        #     mac_game_inequality,
        #     num_steps=150,
        #     step_size=0.06,
        #     sample_width=1,
        #     verbose=True
        # )

        # ghza_qmac_facet_opt_jobs = client.map(ghza_qmac_facet_opt_fn, range(n_workers))
        # ghza_qmac_facet_opt_dicts = client.gather(ghza_qmac_facet_opt_jobs)

        # max_opt_dict = ghza_qmac_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(ghza_qmac_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(ghza_qmac_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = ghza_qmac_facet_opt_dicts[j]

        # scenario = "ghza_qmac_game_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)

        # """
        # GHZA quantum mac facet
        # """
        # client.restart()

        # time_start = time.time()

        # ghza_qmac_facet_opt_fn = optimize_inequality(
        #     ghza_qmac_layers,
        #     postmap9,
        #     mac_facet_inequality,
        #     num_steps=150,
        #     step_size=0.06,
        #     sample_width=1,
        #     verbose=True
        # )

        # ghza_qmac_facet_opt_jobs = client.map(ghza_qmac_facet_opt_fn, range(n_workers))
        # ghza_qmac_facet_opt_dicts = client.gather(ghza_qmac_facet_opt_jobs)

        # max_opt_dict = ghza_qmac_facet_opt_dicts[0]
        # max_score = max(max_opt_dict["scores"])
        # for j in range(1,n_workers):
        #     if max(ghza_qmac_facet_opt_dicts[j]["scores"]) > max_score:
        #         max_score = max(ghza_qmac_facet_opt_dicts[j]["scores"])
        #         max_opt_dict = ghza_qmac_facet_opt_dicts[j]

        # scenario = "ghza_qmac_facet_"
        # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # qnetvo.write_optimization_json(
        #     max_opt_dict,
        #     data_dir + scenario + inequality_tag + datetime_ext,
        # )

        # print("iteration time  : ", time.time() - time_start)
