'''
Model FEC and packet retransmission together
using Markov chain
'''
import math
import json
import pickle
import time
import matplotlib.pyplot as plt
from scipy import special
from multiprocessing import Pool
import argparse
import numpy as np
import logging
import os

beta_list = np.concatenate((np.arange(0, 1, 0.05), np.arange(1, 6, 1)))


class NetworkState:
    rtt = None          # ms
    loss_rate = None    # [0, 1]
    group_delay = None
    def __init__(self) -> None:
        self.group_delay = 0.18

class MarkovModel:
    rtt = None          # ms
    loss_rate = None    # [0, 1]
    group_size : int = None   # [0, 50]
    delay_ddl : int = None    # ms
    max_layer : int = None
    group_delay : float = None

    # {(fec_count, n0, n1) : p01}
    trans_pbl_cache : dict[tuple[int]:float] = None
    comb_cache : dict[tuple:int] = None
    loss_pwr_cache : list[int] = None
    inv_loss_pwr_cache : list[int] = None

    # matrix
    # [delay, ...]
    node_heap : list[float] = None
    # {delay : {(tx_count, data_count, fec_count): probability, ...}, ...}
    node_dict : dict[float:dict[tuple:float]] = None
    miss_pbl = None
    expected_extra_pkts = None

    def __init__(self, loss_rate, c=1e-4, comb_cache=None, trans_pbl_cache=None):
        # self.rtt = int(network_state.rtt)
        self.loss_rate = loss_rate
        self.coeff = c
        self.max_layer = 15 
        self.comb_cache = comb_cache
        self.trans_pbl_cache = trans_pbl_cache
        # self.comb_cache, self.trans_pbl_cache = generate_cache(self.loss_rate, group_size)
        # init loss_pwr_cache and inv_loss_pwr_cache
        cache_size = 330    # TODO: round(frame_size * 6)
        self.loss_pwr_cache = [0] * (cache_size + 1)
        self.inv_loss_pwr_cache = [0] * (cache_size + 1)
        for i in range(0, cache_size + 1):
            self.loss_pwr_cache[i] = self.loss_rate ** i
            self.inv_loss_pwr_cache[i] = (1 - self.loss_rate) ** i

    def __del__(self):
        self.loss_pwr_cache.clear()
        self.inv_loss_pwr_cache.clear()

    def reset_matrix(self, max_layer=None):
        if max_layer is None:
            max_layer = self.max_layer
        self.chain = {}
        for layer in range(max_layer+1):
            chain_layer = {}
            for pkts in range(55 + 1):
                chain_layer[pkts] = {'beta_pkt': -1, 'qoe': -1, 'dmr': -1, 'gd': -1}
            self.chain[layer] = chain_layer

    # return an integer
    def decode_fec_delay(self, batch_data_count, fec_count):
        experience = (batch_data_count + fec_count) * self.group_delay
        # return 0
        return experience

    def get_fec_pkt_num(self, beta_list, tx_count, data_pkt_num):
        return int(math.floor(data_pkt_num * beta_list[tx_count]))

    def comb(self, n, k):
        if self.comb_cache is not None:
            return self.comb_cache[(n, k)]
        else:
            return special.comb(n, k, exact=True)

    def loss_pwr(self, n):
        return self.loss_pwr_cache[n]

    def inv_loss_pwr(self, n):
        return self.inv_loss_pwr_cache[n]

    # calculate transition probability from S(r0, n0) to S(r1, n1)
    def cal_trans_pbl_2(self, beta_list, r0, n0, n1):
        fec_count = self.get_fec_pkt_num(beta_list, r0, n0)
        return self.cal_trans_pbl(fec_count, r0, n0, n1)

    def cal_trans_pbl(self, fec_count, n0, n1):
        # if self.trans_pbl_cache is not None:
        #     return self.trans_pbl_cache[(fec_count, n0, n1)]
        p01 = 0
        # if fec_count > n0:
        #     assert(fec_count % n0 == 0)

        # S(r0, n0) -> S(r1, 0)
        if n1 == 0:
            for i in range(0, fec_count + 1):
                p01 = p01 + \
                    self.comb(fec_count + n0, i) * self.loss_pwr(i) * self.inv_loss_pwr(fec_count + n0 - i)
        # S(r0, n0) -> S(r1, n1)
        elif n1 != 0:
            min_loss_count = int(max(fec_count + 1, n1))
            assert min_loss_count <= n1 + fec_count
            for i in range(min_loss_count, n1 + fec_count + 1):
                assert n1 <= i
                p01 = p01 + \
                    self.comb(fec_count + n0, i) * self.loss_pwr(i) * self.inv_loss_pwr(fec_count + n0 - i) * \
                    self.comb(fec_count, i - n1) * self.comb(n0, n1) / self.comb(fec_count + n0, i)

        return p01

    # packet_arrive_dis = {tx_count : expected_pkt_count, ...}
    packet_arrive_dis = {}
    # group_finish_dis = {tx_count : finish_pbl}
    group_finish_dis = {}

    # calculate the probability of every state
    # the chain depends on the loss rate and frame size
    def build_chain(self, verbose=False):
        self.reset_matrix()
        if verbose:
            print("Frame size: %d, loss rate: %.2f" % (self.frame_size, self.loss_rate))
        # calculation
        # initialize the last layer: the max_layer has determined to violated the deadline
        # therefore the bw loss is 0 (no rate is needed)

        # the layer is counted from the back of the chain, i.e. 0 represents the last layer
        max_frame_size = 55
        self.chain[0][0]['qoe'] = cal_qoe(0, 0, self.coeff)
        for pkts in range(1, max_frame_size + 1):
            self.chain[0][pkts]['qoe'] = cal_qoe(1, 0, self.coeff)
        for layer in range(1, self.max_layer+1):
            self.chain[layer][0]['qoe'] = cal_qoe(0, 0, self.coeff)
            for pkts in range(1, max_frame_size + 1):
                best_qoe = -np.inf
                best_beta_pkt = -1
                for cur_beta_pkt in np.unique(np.round(beta_list * pkts)).astype(int).tolist():  # remove redundant items to accelerate for small pkts
                    cur_qoe = 0
                    for n1 in range(pkts + 1):
                        trans_pbl = self.cal_trans_pbl(cur_beta_pkt, pkts, n1)
                        cur_qoe += trans_pbl * self.chain[layer-1][n1]['qoe']
                    cur_qoe += cal_qoe(0, cur_beta_pkt / self.frame_size, self.coeff)
                    if cur_qoe > best_qoe:
                        best_beta_pkt = cur_beta_pkt
                        best_qoe = cur_qoe
                assert(best_beta_pkt >= 0)
                self.chain[layer][pkts]['beta_pkt'] = best_beta_pkt
                self.chain[layer][pkts]['qoe'] = best_qoe

def cal_qoe(miss_pbl, bw_loss, c=1e-4):
    return - miss_pbl - c * bw_loss


def chain_lookup(loss_rate, frame_size, layer, pkts, coeff):
    # comb_cache, trans_pbl_cache = load_cache(loss_rate)
    # model = MarkovModel(loss_rate=loss_rate, c=coeff, comb_cache=comb_cache, trans_pbl_cache=trans_pbl_cache)
    model = MarkovModel(loss_rate=loss_rate, c=coeff)
    model.frame_size = frame_size
    model.build_chain()
    return model.chain[layer][pkts]['beta_pkt']


def iterate_frame_size(loss_rate, frame_size_list, c, verbose=False):
    frame_size_chains = {}
    # print("calculating model rtt:%dms, loss: %3.2f%%, group_size: %d" % (network_state.rtt, network_state.loss_rate * 100, group_size))
    
    comb_cache, trans_pbl_cache = load_cache(loss_rate)
    model = MarkovModel(loss_rate=loss_rate, c=c, comb_cache=comb_cache, trans_pbl_cache=trans_pbl_cache)
    # TODO: the cache of trans_pbl and loss_pwr
    for frame_size in frame_size_list:
        model.frame_size = frame_size
        model.reset_matrix()
        model.build_chain() # it updates self.chain
        frame_size_chains[frame_size] = model.chain
    return frame_size_chains


def calculate_baseline(network_state: NetworkState, group_size, pkl_fnames, delay_ddl=80, c=1e-4, verbose=False):
    # test: calculate baseline: beta==0
    baseline_model = MarkovModel(network_state, group_size, delay_ddl)
    bl_exp_miss, bl_extra_pkt = baseline_model.build([0] * group_size)
    if verbose:
        with open(pkl_fnames[2], 'a') as f:
            f.write("Baseline:\t alpha: %.2f, beta_0: %.2f, beta_1: %.2f, exp_miss: %.20f, extra_pkt: %.20f\n" % (network_state.loss_rate, 0, 0, bl_exp_miss, bl_extra_pkt))
            f.flush()
            f.close()

def iterate_loss_rate(loss_rate_list, frame_size_list, c=1e-4, verbose=False):
    pool = Pool()
    all_results_list = pool.starmap(iterate_frame_size, [(loss_rate, frame_size_list, c, verbose) for loss_rate in loss_rate_list])
    pool.close()
    pool.join()

    all_results_dict = {}
    for loss_idx in range(len(loss_rate_list)):
        all_results_dict[loss_rate_list[loss_idx]] = all_results_list[loss_idx]
    
    return all_results_dict

'''
==========================================================================
============= For best beta plot at different c and loss_rate ============
================================= STARTS =================================
'''
def restore_best_beta_plot(filename):
    loss_rate_list = None
    best_beta_dict = None
    with open(filename, 'rb') as f:
        loss_rate_list = pickle.load(f)
        best_beta_dict = pickle.load(f)
    return loss_rate_list, best_beta_dict

def calculate_best_beta_plot(loss_rate_list, frame_size_list, pkl_fnames, delay_ddl=80, verbose=False):
    # remove log files
    if verbose:
        for f in pkl_fnames:
            if os.path.exists(f):
                os.remove(f)

    c_list = [1e-4]
    # c_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    comb_cache, trans_pbl_cache = generate_cache(loss_rate_list)

    # init dict to store beta_beta
    
    result_list = iterate_loss_rate(loss_rate_list, frame_size_list, c_list[0], verbose=verbose)

    # store best_beta_dict
    with open(pkl_fnames[0], 'wb') as f:
        pickle.dump(best_beta_dict, f)
        f.flush()
        f.close()
    pickle_to_json(pkl_fnames[0])
    return best_beta_dict

def plot_best_betas(best_beta_dict):
    # plot
    fig, axes = plt.subplots(1, 2)
    for c, best_beta_list in best_beta_dict.items():
        axes[0].plot([b[0] for b in best_beta_list], [b[1][0] for b in best_beta_list], label=str(c))
        axes[1].plot([b[0] for b in best_beta_list], [b[1][1] for b in best_beta_list], label=str(c))

    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$\beta_0$")
    axes[0].set_xlim(0, loss_rate_list[-1])
    axes[0].set_ylim(0, 1)
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$\beta_1$")
    axes[1].set_xlim(0, loss_rate_list[-1])
    axes[1].set_ylim(0, 5)
    plt.legend(title=r"$c$")
    plt.savefig('fec_w_rtx_%d_%dms_%dms.pdf' % (args.group_size, args.rtt, args.ddl))
    plt.show()
'''
================================== ENDS ==================================
============= For best beta plot at different c and loss_rate ============
==========================================================================
'''


def pickle_to_json(binary_filename):
    result = []
    fb = open(binary_filename, 'rb')
    ft = open(binary_filename + '.txt', 'w')
    ft.write('[\n')
    while True:
        # pickle.load(f) = (alpha, ((beta_0, ((beta_1, feature),)),))
        try:
            json.dump(pickle.load(fb), ft, indent=4)
            ft.write(',')
        except EOFError:
            ft.seek(ft.tell() - 1, 0)
            break
    ft.write(']\n')
    fb.close()
    ft.close()

def get_log_filenames(group_size, delay_ddl, network_state: NetworkState):
    pkl_fnames = [
        'best_beta_dict_%dms_%dms.pkl' % (network_state.rtt, delay_ddl),
        'matrix_feature_%dms_%dms.pkl' % (network_state.rtt, delay_ddl),
        'matrix_above_baseline_%dms_%dms.txt' % (network_state.rtt, delay_ddl,),
    ]
    return pkl_fnames


# interface for python simulator to call
# given rtt, remaining time, loss rate, fec group size and QoE coeff
# return (beta_0, beta_1) that makes the highest QoE
def get_best_beta_interface(rtt: int, ddl: int, loss_rate: float, group_size: int=None, coeff: float=1e-4, group_delay=0, data_num=None) -> tuple:
    print("Running model... loss: %f" % loss_rate)
    if loss_rate == 0:
        # loss rate = 0, no FEC needed
        return (0, 0), group_size
    network_state = NetworkState()
    network_state.rtt = rtt
    network_state.loss_rate = loss_rate
    network_state.group_delay = group_delay
    pkl_fnames = get_log_filenames(group_size, ddl, network_state)
    with open('', 'rb') as fb, open('', 'rb') as fm:
        beta_dict = pickle.load(fm)
        block_dict = pickle.load(fb)

    if group_size is None:
        group_size_list = np.arange(5, 51, 5)
        best_beta = None
        best_qoe = -1e9
        best_miss_rate = None
        best_bw_loss = None
        for size in group_size_list:
            beta, miss_rate, bw_loss, qoe = \
                find_best_beta(network_state, size, pkl_fnames, ddl, c=coeff)
            if data_num is not None:
                if size - data_num >= 5:
                    continue
                group_count = data_num / size
                miss_rate = 1 - (1 - miss_rate) ** group_count
                qoe = cal_qoe(miss_rate, bw_loss, coeff)
            if qoe >= best_qoe:
                best_beta = beta
                best_qoe = qoe
                group_size = size
                best_miss_rate = miss_rate
                best_bw_loss = bw_loss
    else:
        best_beta, best_miss_rate, best_bw_loss, best_qoe = \
            find_best_beta(network_state, group_size, pkl_fnames, ddl, coeff)

    logging.info("fec_w_rtx.run_model: Best b0: %.5f, b1:%.5f at alpha: %.5f" % (best_beta[0], best_beta[1], loss_rate))
    return best_beta, best_qoe, best_miss_rate, best_bw_loss, group_size

def generate_comb_cache(max_n=55, max_beta=5) -> dict[tuple[int]:int]:
    cache = {}
    for i in range(0, max_n * (max_beta+1) + 1):
        for j in range(0, max_n * (max_beta+1) + 1):
            if i >= j:
                cache[(i, j)] = special.comb(i, j, exact=True)
    return cache

def generate_trans_pbl_cache(comb_cache, loss_rate_list, max_group_size, max_fec_rate=5) -> dict[dict[tuple:float]]:
    fec_count_list = np.arange(0, max_group_size + 1)
    for fec_rate in range(2, 6):
        fec_count_list = np.concatenate((fec_count_list, fec_count_list * fec_rate))
    fec_count_list = set(fec_count_list)
    # trans_pbl_cache = {loss_rate: {(fec_count, n0, n1): p01, ...}, ...}
    trans_pbl_cache = {}
    network_state = NetworkState()
    network_state.rtt = 10  # does not matter
    network_state.group_delay = 0.18  # does not matter
    for loss_rate in loss_rate_list:
        print("Calculating loss rate %.2f%%" % (loss_rate * 100))
        network_state.loss_rate = loss_rate
        loss_rate = round(loss_rate * 100) / 100
        model = MarkovModel(loss_rate, comb_cache=comb_cache)   # TODO: no coeff!
        model.frame_size = max_group_size
        trans_pbl_cache[loss_rate] = {}
        for fec_count in fec_count_list:
            for n0 in range(1, max_group_size + 1):
                for n1 in range(0, max_group_size + 1):
                    if n0 < n1:
                        continue
                    if fec_count > n0 and (fec_count % n0 != 0 or fec_count > n0 * max_fec_rate):
                        continue
                    trans_pbl_cache[loss_rate][(fec_count, n0, n1)] = \
                        model.cal_trans_pbl(fec_count, n0, n1)
    return trans_pbl_cache


comb_cache_fname = "./cache/comb_cache.pkl"
trans_pbl_cache_fname = "./cache/trans_pbl_cache_%.2f%%.pkl"
# return comb_cache, trans_pbl_cache
def generate_cache(loss_rate_list, max_group_size=55):
    if not os.path.exists("./cache"):
        os.mkdir("./cache")
    # cache for combination calculation
    if not os.path.exists(comb_cache_fname):
        print("Generating combination cache...")
        comb_cache = generate_comb_cache()
        with open(comb_cache_fname, "wb") as f:
            pickle.dump(comb_cache, f)
            f.close()
    else:
        with open(comb_cache_fname, 'rb') as f:
            comb_cache = pickle.load(f)
            f.close()
    # cache for transition probability calculation in Markov Chain
    # trans_pbl_cache = {loss_rate: {(fec_count, n0, n1): p01, ...}, ...}
    if not os.path.exists(trans_pbl_cache_fname % (loss_rate_list[0] * 100)):
        print("Generating transition probability cache...")
        trans_pbl_cache = generate_trans_pbl_cache(comb_cache, loss_rate_list, max_group_size)
        for loss_rate, result in trans_pbl_cache.items():
            with open(trans_pbl_cache_fname % (loss_rate * 100), "wb") as f:
                pickle.dump(result, f)
                f.close()


def load_cache(loss_rate):
    with open(comb_cache_fname, 'rb') as f:
        comb_cache = pickle.load(f)
        
    with open(trans_pbl_cache_fname % (loss_rate * 100), 'rb') as f:
        trans_pbl_cache = pickle.load(f)

    return comb_cache, trans_pbl_cache


# prepare model results for online usage
# given ddl and QoE coeff
# iterate over rtt, loss rate, group size
# store pickle file and C code
def generate_best_beta_results(coeff: float=1e-4, verbose=False):
    # loss rate ranges from 0% to 50%
    loss_rate_start = 0.0
    loss_rate_end = 0.50
    loss_rate_itvl = 0.01
    loss_rate_list = np.linspace(loss_rate_start, loss_rate_end, int(round((loss_rate_end - loss_rate_start) / loss_rate_itvl)) + 1).tolist()
    print("loss_rate ranges from %.2f%% to %.2f%%" % (loss_rate_start * 100, loss_rate_end * 100))
    print(loss_rate_list)
    # group size ranges from 5 to 55
    frame_size_start = 5
    frame_size_end = 55
    frame_size_itvl = 5
    frame_size_list = np.linspace(frame_size_start, frame_size_end,int(round((frame_size_end - frame_size_start) / frame_size_itvl)) + 1, dtype=int).tolist()
    print("frame_size ranges from %d to %d" % (frame_size_start, frame_size_end))
    print(frame_size_list)

    total_num = len(loss_rate_list) * len(frame_size_list) 
    print("%d jobs" % total_num)

    generate_cache(loss_rate_list, frame_size_end)

    print("Calculating...")
    result_list = iterate_loss_rate(loss_rate_list, frame_size_list, coeff, verbose)
    print("Ends at " + str(time.time()))

    pickle_filename = "model_result_%.0e.pkl" % (coeff)
    if os.path.exists(pickle_filename):
        os.remove(pickle_filename)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(result_list, f)
        f.close()
    pickle_to_json(pickle_filename)
    return result_list


def block_iteration(loss_rate, chain_list, frame_size_list, ddl_list, rtt_list, rdisp_list):
    print("block search loss rate: %.2f" % loss_rate)
    result_loss_rate = {}
    for frame_size in frame_size_list:
        cur_chain = chain_list[loss_rate][frame_size]
        result_frame_size = {}
        for ddl in ddl_list:
            result_ddl = {}
            for rtt in rtt_list:
                result_rtt = {}
                for rdisp in rdisp_list:
                    if rdisp == 0:
                        result_rtt[rdisp] = frame_size
                        continue
                    best_block_size = frame_size
                    base_layer = int(np.floor((ddl - frame_size * rdisp) / rtt))
                    if base_layer < 1:  # there is no chance to deliver even one packet
                        continue
                    best_qoe = cur_chain[base_layer][frame_size]['qoe']
                    max_additional_block = max(0, int(np.floor((ddl - (base_layer + 1) * rtt) / rdisp)))
                    # the maximum possible block size to enjoy a new transmission chance
                    for block_size in range(1, max_additional_block + 1):
                        cur_qoe = 0
                        add_layer_block_cnt = int(np.floor((ddl - (base_layer + 1) * rtt) / (block_size * rdisp)))
                        cur_qoe += add_layer_block_cnt * cur_chain[base_layer+1][block_size]['qoe']
                        base_layer_block_cnt = int(np.ceil(frame_size / block_size)) - add_layer_block_cnt
                        cur_qoe += base_layer_block_cnt * cur_chain[base_layer][block_size]['qoe']
                        if cur_qoe > best_qoe:
                            best_qoe = cur_qoe
                            best_block_size = block_size
                    result_rtt[rdisp] = best_block_size
                result_ddl[rtt] = result_rtt
            result_frame_size[ddl] = result_ddl
        result_loss_rate[frame_size] = result_frame_size
    return result_loss_rate


def generate_best_block_results(chain_list, coeff):
    # loss rate ranges from 0% to 50%
    loss_rate_start = 0.0
    loss_rate_end = 0.50
    loss_rate_itvl = 0.01
    loss_rate_list = np.linspace(loss_rate_start, loss_rate_end, int(round((loss_rate_end - loss_rate_start) / loss_rate_itvl)) + 1).tolist()
    print("loss_rate ranges from %.2f%% to %.2f%%" % (loss_rate_start * 100, loss_rate_end * 100))
    print(loss_rate_list)

    # group size ranges from 5 to 55
    frame_size_start = 5
    frame_size_end = 55
    frame_size_itvl = 5
    frame_size_list = list(range(frame_size_start, frame_size_end + 1, frame_size_itvl))
    print("frame_size ranges from %d to %d" % (frame_size_start, frame_size_end))
    print(frame_size_list)

    # ddl from 20 to 140
    ddl_start = 20
    ddl_end = 140
    ddl_step = 20
    ddl_list = list(range(ddl_start, ddl_end + 1, ddl_step))

    # rtt from 6 to 80
    rtt_start = 10
    rtt_end = 80
    rtt_step = 2
    rtt_list = list(range(rtt_start, rtt_end + 1, rtt_step))

    # dispersion rate from 0ms/pkt to 1ms/pkt
    rdisp_start = 0
    rdisp_end = 1
    rdisp_step = 0.02
    rdisp_list = np.linspace(rdisp_start, rdisp_end, int(round((rdisp_end - rdisp_start) / rdisp_step)) + 1).tolist()

    result_block = {}
    pool = Pool()
    result_loss_rate_list = pool.starmap(block_iteration, [
            (loss_rate, chain_list, frame_size_list, ddl_list, rtt_list, rdisp_list) for loss_rate in loss_rate_list
    ])
    pool.close()
    pool.join()  
    for loss_rate_idx in range(len(loss_rate_list)):
        result_block[loss_rate_list[loss_rate_idx]] = result_loss_rate_list[loss_rate_idx]

    pickle_filename = "block_result_%.0e.pkl" % (coeff)
    if os.path.exists(pickle_filename):
        os.remove(pickle_filename)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(result_block, f)
        f.close()
    pickle_to_json(pickle_filename)


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtt', type=int, default=20)
    parser.add_argument('--ddl', type=int, default=100)
    parser.add_argument('--group-size', type=int, default=None)
    parser.add_argument('--group-delay', type=float, default=1)
    parser.add_argument('--loss', type=float, default=None)
    parser.add_argument('--coeff', type=float, default=1e-4)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--beta2', type=float, default=None)
    parser.add_argument('--data-num', type=int, default=None)
    parser.add_argument('--remaining-time', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--pkts', type=int)
    parser.add_argument('--func', type=str, default='offline_data', choices=['model', 'best', 'plot', 'offline_data'])
    args = parser.parse_args()

    logging.basicConfig(filename='fec_model.log', format='%(asctime)s %(message)s', level=logging.ERROR)

    if args.verbose:
        import pickle
        import json
        import matplotlib.pyplot as plt
        import os

    if args.func == 'offline_data':
        print("Generating offline data...")
        print("coeff: %.0e" % (args.coeff))
        assert args.coeff is not None
        result_list = generate_best_beta_results(args.coeff)
        generate_best_block_results(result_list, args.coeff)
    elif args.func == 'best':
        print("Looking for best FEC rate...")
        assert args.rtt is not None
        assert args.loss is not None
        assert args.remaining_time is not None
        assert args.pkts is not None
        assert args.frame_size is not None
        assert args.group_delay is not None
        # return best betas with group-size provided
        layer = int(np.floor(args.remaining_time / args.rtt))
        best_beta = chain_lookup(loss_rate=args.loss, frame_size=args.frame_size, layer=layer, pkts=args.pkts, coeff=args.coeff)
        print("best beta: %d" % best_beta)
        # best_beta, best_qoe, best_miss_rate, best_bw_loss, group_size = \
        #     get_best_beta_interface(args.rtt, args.ddl, args.loss, args.group_size, args.coeff, args.group_delay, data_num=args.data_num)
        # print("best beta: %.10f, %.10f, group size: %d, ddl miss rate: %.10f, bw loss rate: %.10f, best qoe: %f" % (best_beta[0], best_beta[1], group_size, best_miss_rate, best_bw_loss, best_qoe))
    elif args.func == 'plot':
        print("Plotting best FEC rates at different loss rates and coefficients...")
        assert args.rtt is not None
        assert args.group_size is not None
        assert args.verbose is not None
        assert args.ddl is not None
        network_state = NetworkState()
        network_state.rtt = args.rtt
        network_state.group_delay = args.group_delay
        pkl_fnames = get_log_filenames(args.group_size, args.ddl, network_state)
        loss_rate_list = np.linspace(0.01, 0.8, 80)
        # loss_rate_list = [0.6]
        best_beta_dict = calculate_best_beta_plot(loss_rate_list, network_state, args.group_size, pkl_fnames, args.ddl, args.verbose)
        # print(best_beta_dict)
        plot_best_betas(best_beta_dict)
    elif args.func == 'model':
        assert args.rtt is not None
        assert args.loss is not None
        assert args.pkts is not None
        assert args.ddl is not None
        assert args.beta is not None

        max_layer = int(np.floor(args.ddl / args.rtt))
        beta_list = [args.beta] * (max_layer+1)
        if args.beta2 is not None:
            beta_list[max_layer-1] = args.beta2

        # comb_cache, trans_pbl_cache = load_cache(args.loss)
        # model = MarkovModel(loss_rate=args.loss, c=args.coeff, comb_cache=comb_cache, trans_pbl_cache=trans_pbl_cache)
        model = MarkovModel(loss_rate=args.loss, c=args.coeff)
        model.frame_size = args.pkts
        model.reset_matrix(max_layer)

        model.chain[0][0]['dmr'] = 0
        model.chain[0][0]['gd'] = 0
        model.chain[0][0]['qoe'] = cal_qoe( model.chain[0][0]['dmr'], model.chain[0][0]['gd'], model.coeff)
        for pkts in range(1, model.frame_size + 1):
            model.chain[0][pkts]['dmr'] = 1
            model.chain[0][pkts]['gd'] = 0
            model.chain[0][pkts]['qoe'] = cal_qoe(model.chain[0][pkts]['dmr'], model.chain[0][pkts]['gd'], model.coeff)
        for layer in range(1, max_layer+1):
            model.chain[layer][0]['dmr'] = 0
            model.chain[layer][0]['gd'] = 0
            model.chain[layer][0]['qoe'] = cal_qoe(0, 0, model.coeff)
            for pkts in range(1, model.frame_size + 1):
                cur_dmr = 0
                for n1 in range(pkts + 1):
                    trans_pbl = model.cal_trans_pbl(round(beta_list[layer] * pkts), pkts, n1)
                    cur_dmr += trans_pbl * model.chain[layer-1][n1]['dmr']
                model.chain[layer][pkts]['dmr'] = cur_dmr
            
        print("beta %.3f max layer %d deadline miss rate: %.10f" % (args.beta, max_layer, model.chain[max_layer][args.pkts]['dmr']))
        # print(model.chain)
        # cur_qoe += cal_qoe(0, args.beta_pkt / model.frame_size, model.coeff)

        # print("At alpha: %.2f, beta_0: %.2f, beta_1: %.2f, miss_rate: %.10f, bw_loss_rate: %.10f, qoe: %f" % (args.loss, args.beta0, args.beta1, miss_pbl, extra_pkt / (args.group_size + extra_pkt), cal_qoe(miss_pbl, extra_pkt / (args.group_size + extra_pkt), args.coeff)))

