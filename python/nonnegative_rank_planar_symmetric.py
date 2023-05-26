import numpy as np
from datetime import datetime
import json

from sklearn.decomposition import NMF

from dask.distributed import Client


def planar_sym_states(n):
    return [[np.cos(np.pi * i/n), np.sin(np.pi * i/n)] for i in range(n)]

def planar_sym_povm(n):
    return list(map(
        lambda x: np.outer(x,np.conjugate(x))*2/n,
        planar_sym_states(n)
    ))

def planar_sym_behavior(n):
    states = planar_sym_states(n)
    z_round = lambda x: 0 if np.isclose(x, 0, atol=1e-9) else x 
    return np.array([
        [
            z_round(np.power(np.inner(states[y], states[x]), 2) * 2/n)
            for x in range(n)
        ]
        for y in range(n)
    ])

def nmf_fn(M, k, **kwargs):
    def nmf(placeholder):
        model = NMF(k, **kwargs)
        A = model.fit_transform(M)
        B = model.components_

        return A, B

    return nmf

def certify_nonnegative_rank(M, n_workers=5, n_jobs=10, n_threads_per_worker=None):

    lb = int(np.linalg.matrix_rank(M))
    ub = int(min(M.shape))

    k = int(lb + np.floor((ub - lb)/2))

    data_dict = {
        "k": None,
        "M_approx": None,
        "A": None,
        "B": None,
        "M": M.tolist(),
        "score": None,
    }

    n_threads_per_worker = n_threads_per_worker if n_threads_per_worker else np.ceil(n_jobs/n_workers)
    client = Client(processes=True, n_workers=n_workers, threads_per_worker=n_threads_per_worker)

    while lb < ub:
        print("lb <= k <= ub : ", lb, " <= ", k, " <= ", ub)
        
        nmf = nmf_fn(M, k, max_iter=300000, tol=1e-18, init="random")

        nmf_jobs = client.map(nmf, range(n_jobs))
        nmf_results = client.gather(nmf_jobs)

        M_approx_list = [A@B for A, B in nmf_results] 
        D_list = [np.abs(M - M_approx) for M_approx in M_approx_list]
        scores = [np.sum(D)/(2*D.shape[1]) for D in D_list]        

        print("scores = ", scores)
        min_id = np.argmin(scores)

        score = scores[min_id]

        print("score : ", score)

        if np.isclose(score, 0, atol=1e-9):
            ub = int(k)
            k = int(lb + np.floor((k - lb)/2))
            data_dict["M_approx"] = M_approx_list[min_id].tolist()
            A, B = nmf_results[min_id]
            data_dict["A"] = A.tolist()
            data_dict["B"] = B.tolist()
            data_dict["score"] = score
        else:
            lb = int(k + 1)
            k = int(k + np.ceil((ub - k)/2))

    print("lb <= k <= ub : ", lb, " <= ", k, " <= ", ub)
    data_dict["k"] = k
    if k == min(M.shape):
        data_dict["M_approx"] = M.tolist()
        data_dict["score"] = 0
        if M.shape[0] > M.shape[1]:
            data_dict["B"] = np.eye(k).tolist()
            data_dict["A"] = M.tolist()
        else:
            data_dict["B"] = M.tolist()
            data_dict["A"] = np.eye(k).tolist()
        
    return data_dict

if __name__ == "__main__":

    data_path = "data/nonnegative_rank/qubit_planar_symmetric/"

    for n in range(7, 33):
        datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        P = planar_sym_behavior(n)
        data_dict = certify_nonnegative_rank(P)
        filename = data_path + "n_" + str(n) + "_" + datetime_ext
        with open(filename + ".json", "w") as file:
            file.write(json.dumps(data_dict, indent=2))

