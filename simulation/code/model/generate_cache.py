import numpy as np
import pickle

from .fec_w_rtx import run_model


if __name__ == "__main__":
    alg_cache_file = "./code/model/alg_cache.pkl"
    rtt = 20
    bandwidth = 30
    delay_ddl = 80
    rtt = int(rtt)
    alg_cache_dict = {}
    for loss_rate in np.arange(0, 1.01, 0.01):
        bandwidth = round(bandwidth, 2)
        delay_ddl = int(delay_ddl)
        loss_rate = round(loss_rate, 2)
        key = (rtt, bandwidth, delay_ddl, loss_rate)
        alg_cache_dict[key] = run_model(rtt, bandwidth, delay_ddl, loss_rate)


    with open(alg_cache_file, 'wb') as f:
        pickle.dump(alg_cache_dict, f)
        f.close()