'''
Model FEC and packet retransmission together
using Markov chain
'''
import math
import multiprocessing
from operator import mod
import pickle
import time
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import nonzero, product
from scipy import special
from multiprocessing import Pool
import argparse
import numpy as np
import logging
import os
import itertools
import heapq
import cProfile
import copy
# from sklearn import tree
# from sklearn_porter import Porter


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

    def __init__(self, network_state: NetworkState, group_size : int, delay_ddl=80, comb_cache=None, trans_pbl_cache=None):
        self.rtt = int(network_state.rtt)
        self.loss_rate = network_state.loss_rate
        self.group_delay = network_state.group_delay
        self.group_size = int(group_size)
        self.delay_ddl = int(delay_ddl) - self.rtt / 2
        self.max_layer = int(math.floor((delay_ddl - self.rtt) / self.rtt)) + 2
        self.node_heap = []
        self.node_dict = {}
        self.reset_matrix()
        self.comb_cache = comb_cache
        self.trans_pbl_cache = trans_pbl_cache
        # self.comb_cache, self.trans_pbl_cache = generate_cache(self.loss_rate, group_size)
        # init loss_pwr_cache and inv_loss_pwr_cache
        cache_size = round(group_size * 6)
        self.loss_pwr_cache = [0] * (cache_size + 1)
        self.inv_loss_pwr_cache = [0] * (cache_size + 1)
        for i in range(0, cache_size + 1):
            self.loss_pwr_cache[i] = self.loss_rate ** i
            self.inv_loss_pwr_cache[i] = (1 - self.loss_rate) ** i

    def __del__(self):
        # print("calculating model rtt:%dms, loss: %3.2f%%, group_size: %d" % (self.rtt, self.loss_rate * 100, self.group_size)
        self.node_heap.clear()
        self.node_dict.clear()
        self.loss_pwr_cache.clear()
        self.inv_loss_pwr_cache.clear()

    def reset_matrix(self):
        self.miss_pbl = 0
        self.expected_extra_pkts = - self.group_size
        # init node_heap
        self.node_heap.clear()
        self.node_dict.clear()

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
        if self.trans_pbl_cache is not None:
            return self.trans_pbl_cache[(fec_count, n0, n1)]
        p01 = 0
        if fec_count <= n0:
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
        elif fec_count > n0:
            # FEC rate should be integer
            assert fec_count % n0 == 0
            fec_rate = int(round(fec_count / n0))
            p01 = self.comb(n0, n1) * self.loss_pwr((fec_rate + 1) * n1) * (1 - self.loss_pwr(fec_rate + 1)) ** (n0 - n1)

        return p01

    def build_2(self, beta_0, beta_1, check=False, verbose=False):
        beta_list = [beta_0] + [beta_1] * (self.max_layer)
        return self.build(beta_list, check, verbose)

    # packet_arrive_dis = {tx_count : expected_pkt_count, ...}
    packet_arrive_dis = {}
    # group_finish_dis = {tx_count : finish_pbl}
    group_finish_dis = {}

    # calculate the probability of every state
    def build(self, beta_list, check=False, verbose=False):
        self.reset_matrix()
        if verbose:
            print("Group size: %d" % self.group_size)
        # calculation
        logging.info("fec_w_rtx: Building markov matrix with loss_rate: %3.3f" % (self.loss_rate, ))
        # init node
        # push time 0 into heap
        heapq.heappush(self.node_heap, 0)
        # push the first transmission (the very first node) into dict
        self.node_dict[0] = {}
        self.node_dict[0][(0, self.group_size)] = 1
        while True:
            try:
                earliest_time : float = heapq.heappop(self.node_heap)
            except IndexError as e:
                # run out of self.node_heap
                break
            if self.node_dict.get(earliest_time) is None:
                continue
            for (tx_count, data_count), pbl in self.node_dict[earliest_time].items():
                fec_count = self.get_fec_pkt_num(beta_list, tx_count, data_count)
                # multiply packet number with the node's probability
                self.expected_extra_pkts += pbl * (data_count + fec_count)
                # the time that the group of packets arrive at client
                rcv_time = earliest_time  + self.rtt / 2 + self.decode_fec_delay(data_count, fec_count)
                # the time that server receives the next retransmission request
                next_rtx_time = rcv_time + self.rtt / 2
                # # round to 0.5 ms
                rcv_time = round(rcv_time)
                next_rtx_time = round(next_rtx_time)

                if rcv_time > self.delay_ddl:
                    # ddl missed, all packets are useless
                    self.miss_pbl += pbl
                    continue

                # packet_arrive_dis
                for next_data_count in range(0, data_count + 1):
                    # there will be more retransmissions
                    # next_data_count == num of lost packets
                    trans_pbl = self.cal_trans_pbl(fec_count, data_count, next_data_count)
                    if self.packet_arrive_dis.get(tx_count) is None:
                        self.packet_arrive_dis[tx_count] = 0
                    self.packet_arrive_dis[tx_count] += \
                        pbl * self.cal_trans_pbl(fec_count, data_count, next_data_count) * (data_count - next_data_count)

                # group_finish_dis
                if self.group_finish_dis.get(tx_count + 1) is None:
                    self.group_finish_dis[tx_count + 1] = 0
                self.group_finish_dis[tx_count + 1] += \
                    pbl * self.cal_trans_pbl(fec_count, data_count, 0)

                if next_rtx_time > self.delay_ddl:
                    # packets can be received by client, but no more retransmissions
                    # in this case:
                    #   probability of the next level of nodes should be counted
                    #   but the next level of nodes shouldn't exist in the model
                    self.miss_pbl += pbl * (1 - self.cal_trans_pbl(fec_count, data_count, 0))
                    continue


                for next_data_count in range(1, data_count + 1):
                    # there will be more retransmissions
                    # next_data_count == num of lost packets
                    trans_pbl = self.cal_trans_pbl(fec_count, data_count, next_data_count)
                    if self.node_dict.get(next_rtx_time) is None:
                        self.node_dict[next_rtx_time] = {}
                        heapq.heappush(self.node_heap, next_rtx_time)
                    if self.node_dict[next_rtx_time].get((tx_count + 1, next_data_count)) is None:
                        self.node_dict[next_rtx_time][(tx_count + 1, next_data_count)] = 0
                    self.node_dict[next_rtx_time][(tx_count + 1,next_data_count)] += pbl * trans_pbl

            del self.node_dict[earliest_time]
        if verbose:
            print("packet_arrive_distribution" + str(self.packet_arrive_dis))
            print("group_finish_distribution" + str(self.group_finish_dis))

        return self.miss_pbl, max(self.expected_extra_pkts, 0)


def cal_qoe(miss_pbl, bw_loss, c=1e-4):
    return - miss_pbl + c * (1 - bw_loss)


def build_markov_process(network_state, group_size, delay_ddl, beta_list : list[tuple], comb_cache=None, trans_pbl_cache=None):
    # print("calculating model rtt:%dms, loss: %3.2f%%, group_size: %d" % (network_state.rtt, network_state.loss_rate * 100, group_size))
    model = MarkovModel(network_state, group_size, delay_ddl,
        comb_cache=comb_cache, trans_pbl_cache=trans_pbl_cache)
    (beta_0, beta_1) = beta_list
    # miss_pbl, extra_pkt = cProfile.runctx('model.build_2(beta_0, beta_1)', globals(), locals())
    miss_pbl, extra_pkt = model.build_2(beta_0, beta_1)
    return (beta_0, beta_1, miss_pbl, extra_pkt)

def calculate_baseline(network_state: NetworkState, group_size, pkl_fnames, delay_ddl=80, c=1e-4, verbose=False):
    # test: calculate baseline: beta==0
    baseline_model = MarkovModel(network_state, group_size, delay_ddl)
    bl_exp_miss, bl_extra_pkt = baseline_model.build([0] * group_size)
    if verbose:
        with open(pkl_fnames[2], 'a') as f:
            f.write("Baseline:\t alpha: %.2f, beta_0: %.2f, beta_1: %.2f, exp_miss: %.20f, extra_pkt: %.20f\n" % (network_state.loss_rate, 0, 0, bl_exp_miss, bl_extra_pkt))
            f.flush()
            f.close()
    # test: comparison with baseline
    # for (beta_0, result_list) in result_list:
    #     for (beta_1, miss_pbl, extra_pkt) in result_list:
    #         if miss_pbl < bl_exp_miss and extra_pkt < bl_extra_pkt:
    #             if verbose:
    #                 with open(pkl_fnames[2], 'a') as f:
    #                     f.write("\t\t\t alpha: %.2f, beta_0: %.2f, beta_1: %.2f, exp_miss: %.20f, extra_pkt: %.20f\n" % (network_state.loss_rate, beta_0, beta_1, miss_pbl, bl_exp_miss))
    #                     f.flush()
    #                     f.close()


# interface
def find_best_beta_3(params):
    start_time = time.time()
    rtt, loss, group_size, group_delay, pkl_fnames, delay_ddl, c = params
    print("Calculating model rtt:%dms, loss: %3.2f%%, group_size: %d, group_delay: %.2fms" % (rtt, loss * 100, group_size, group_delay))
    # result = cProfile.runctx('find_best_beta_2(rtt, loss, group_size, group_delay, pkl_fnames, delay_ddl, c)', globals(), locals())
    result = find_best_beta_2(rtt, loss, group_size, group_delay, pkl_fnames, delay_ddl, c)
    print("Model rtt:%dms, loss: %3.2f%%, group_size: %d, group_delay: %.2fms calculated in %.2fs" % (rtt, loss * 100, group_size, group_delay, time.time() - start_time))
    return result

# interface
def find_best_beta_2(rtt, loss, group_size, group_delay, pkl_fnames, delay_ddl=80, c=1e-4, verbose=False):
    network_state = NetworkState()
    network_state.rtt = rtt
    network_state.loss_rate = loss
    network_state.group_delay = group_delay
    best_beta, best_miss_rate, best_bw_loss, best_qoe = \
        find_best_beta(network_state, group_size, pkl_fnames, delay_ddl, c, verbose)
    return ((rtt, loss, group_delay), group_size, delay_ddl, c, best_beta, best_miss_rate, best_bw_loss, best_qoe)

def find_best_beta(network_state: NetworkState, group_size, pkl_fnames, delay_ddl=80, c=1e-4, verbose=False):
    # single-threaded
    # iterate over those beta values
    b0_list = np.linspace(0, 1, 51)
    b1_list = np.linspace(0, 5, 6)
    logging.info("fec_w_rtx: Building markov matrix with loss_rate: %.5f" % (network_state.loss_rate))

    comb_cache, trans_pbl_cache = load_cache(network_state.loss_rate)

    # result_lists = [(beta_0, beta_1, miss_pbl, extra_pkt), ...]
    # single thread
    result_list = itertools.starmap(
        build_markov_process, [
            (network_state, group_size, delay_ddl, beta_list, comb_cache, trans_pbl_cache) for beta_list in itertools.product(b0_list, b1_list)
        ]
    )

    # now we have results for every (beta_0, beta_1) pair given alpha
    # store it
    if verbose:
        with open(pkl_fnames[1], 'ab') as f:
            pickle.dump((network_state.loss_rate, result_list), f)
            f.flush()
            f.close()

    # calculate best beta pair at given alpha
    # calculate best beta pair at given qoe model (c)
    best_qoe = -1e10
    best_beta = None
    best_miss_rate = None
    best_bw_loss = None
    for (beta_0, beta_1, miss_pbl, extra_pkt) in result_list:
        # print(network_state.loss_rate, beta_0, beta_1, miss_pbl, extra_pkt)
        # given alpha, for every (beta_0, beta_1)
        current_qoe = cal_qoe(miss_pbl, extra_pkt / (extra_pkt + group_size), c)
        if current_qoe > best_qoe:
            best_qoe = current_qoe
            best_beta = (beta_0, beta_1)
            best_miss_rate = miss_pbl
            best_bw_loss = extra_pkt / (extra_pkt + group_size)
    assert best_beta is not None
    del result_list

    return best_beta, best_miss_rate, best_bw_loss, best_qoe

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
        f.close()
    return loss_rate_list, best_beta_dict

def calculate_best_beta_plot(loss_rate_list, network_state: NetworkState, group_size, pkl_fnames, delay_ddl=80, verbose=False):
    # remove log files
    if verbose:
        for f in pkl_fnames:
            if os.path.exists(f):
                os.remove(f)

    c_list = [1e-4]
    # c_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    comb_cache, trans_pbl_cache = generate_cache(loss_rate_list, 45)

    # init dict to store beta_beta
    pool = multiprocessing.Pool()
    # result_list = [((rtt, loss, group_delay), group_size, delay_ddl, c, (beta_0, beta_1), best_miss_rate, best_bw_loss, best_qoe), ...]
    result_list = pool.starmap(
        find_best_beta_2, [
            (network_state.rtt, loss_rate, group_size, network_state.group_delay, None, delay_ddl, c, False, comb_cache, trans_pbl_cache[round(loss_rate * 100) / 100])
            for (c, loss_rate) in itertools.product(c_list, loss_rate_list)
        ]
    )
    pool.close()
    pool.join()


    result_list = [
        ((loss, c), (beta_0, beta_1))
        for (rtt, loss, group_delay), group_size, delay_ddl, c, (beta_0, beta_1), best_miss_rate, best_bw_loss, best_qoe in result_list
    ]
    # result_list = [((loss, c), (beta_0, beta_1)), ...]

    best_beta_dict = {}
    for c in c_list:
        best_beta_dict[c] = []
    for (loss, c), (beta_0, beta_1) in result_list:
        best_beta_dict[c].append((loss, (beta_0, beta_1)))

    # store best_beta_dict
    if verbose:
        with open(pkl_fnames[0], 'wb') as f:
            pickle.dump(best_beta_dict, f)
            f.flush()
            f.close()
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
# given rtt, deadline, loss rate, fec group size and QoE coeff
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

def generate_comb_cache(max_n=50, max_beta=5) -> dict[tuple[int]:int]:
    cache = {}
    for i in range(0, max_n * max_beta + 1):
        for j in range(0, max_n * max_beta + 1):
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
        model = MarkovModel(network_state, max_group_size, comb_cache=comb_cache)
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
        f.close()
    with open(trans_pbl_cache_fname % (loss_rate * 100), 'rb') as f:
        trans_pbl_cache = pickle.load(f)
        f.close()

    return comb_cache, trans_pbl_cache


# prepare model results for online usage
# given ddl and QoE coeff
# iterate over rtt, loss rate, group size
# store pickle file and C code
def generate_best_beta_results(ddl: int, coeff: float=1e-4):
    print("Generating all model results: ddl %d ms" % ddl)
    # rtt ranges from 6ms to 80ms
    rtt_start = 6
    rtt_end = 80
    rtt_itvl = 2
    rtt_list = np.linspace(rtt_start, rtt_end, int(round((rtt_end - rtt_start) / rtt_itvl)) + 1)
    print("rtt ranges from %dms to %dms" % (rtt_start, rtt_end))
    print(rtt_list)
    # loss rate ranges from 0% to 50%
    loss_rate_start = 0.0
    loss_rate_end = 0.50
    loss_rate_itvl = 0.01
    loss_rate_list = np.linspace(loss_rate_start, loss_rate_end, int(round((loss_rate_end - loss_rate_start) / loss_rate_itvl)) + 1)
    print("loss_rate ranges from %.2f%% to %.2f%%" % (loss_rate_start * 100, loss_rate_end * 100))
    print(loss_rate_list)
    # group size ranges from 5 to 55
    group_size_start = 5
    group_size_end = 55
    group_size_itvl = 5
    group_size_list = np.linspace(group_size_start, group_size_end,int(round((group_size_end - group_size_start) / group_size_itvl)) + 1)
    print("group_size ranges from %d to %d" % (group_size_start, group_size_end))
    print(group_size_list)
    # group delay from 0.0ms ~ 0.5ms (per packet)
    group_delay_start = 0.0
    group_delay_end = 1
    group_delay_itvl = 0.04
    group_delay_list = np.linspace(group_delay_start, group_delay_end, int
    (round((group_delay_end - group_delay_start) / group_delay_itvl)) + 1)
    print("group_delay ranges from %.1fms/pkt to %.1fms/pkt" % (group_delay_start, group_delay_end))
    print(group_delay_list)

    total_num = \
        int((rtt_end + rtt_itvl - rtt_start) / rtt_itvl) * \
        int((loss_rate_end + loss_rate_itvl - loss_rate_start) / loss_rate_itvl) * \
        int((group_size_end + group_size_itvl - group_size_start) / group_size_itvl) * \
        int((group_delay_end + group_delay_itvl - group_delay_start) / group_delay_itvl)

    total_num = \
        len(rtt_list) * len(loss_rate_list) * len(group_size_list) * len(group_delay_list)

    print("%d jobs" % total_num)

    generate_cache(loss_rate_list, group_size_end)

    '''
    there will be 39 * 51 * 11 * 20 = 537,580 results
    minimum storage: 537.58K * 2 Byte = 1.07MB
    '''

    print("Calculating...")
    # find_best_beta_3((rtt_list[0], loss_rate_list[0], group_size_list[0], group_delay_list[0], None, ddl, coeff))
    pool = multiprocessing.Pool()
    # result_list = [((rtt, loss, group_delay), group_size, delay_ddl, c, (beta_0, beta_1), best_miss_rate, best_bw_loss), ...]
    result_list = pool.imap(
        find_best_beta_3, [
            (rtt, loss_rate, group_size, group_delay, None, ddl, coeff)
            for (rtt, loss_rate, group_size, group_delay) in itertools.product(rtt_list, loss_rate_list, group_size_list, group_delay_list)
        ],
        chunksize=20
    )
    pool.close()
    pool.join()

    print("Ends at " + str(time.time()))

    result_list = [
        [[round(rtt), round(loss * 100) / 100, round(group_delay * 20) / 20, round(group_size)], [round(math.floor(group_size * beta_0)), round(beta_1)]]
        for (rtt, loss, group_delay), group_size, delay_ddl, c, (beta_0, beta_1), best_miss_rate, best_bw_loss, best_qoe in result_list
    ]
    # result_list = [((rtt, loss, group_delay, group_size), (count_0, beta_1)), ...]


    pickle_filename = "model_result_%d_%.0e.pkl" % (ddl, coeff)
    if os.path.exists(pickle_filename):
        os.remove(pickle_filename)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(result_list, f)
        f.close()

    # # train!
    # print("Start training at " + str(time.time()))
    # clf_beta_0 = tree.DecisionTreeClassifier()
    # clf_beta_1 = tree.DecisionTreeClassifier()
    # clf_beta_0.fit([network for (network, result) in result_list], [result[0] for (network, result) in result_list])
    # clf_beta_0.fit([network for (network, result) in result_list], [result[1] for (network, result) in result_list])

    # porter0 = Porter(clf_beta_0, language='c')
    # output0 = porter0.export(embed_data=True)
    # print(output0)
    # porter1 = Porter(clf_beta_0, language='c')
    # output1 = porter1.export(embed_data=True)
    # print(output1)

    # with open("beta_0_tree.txt", 'w+') as f:
    #     f.write(output0)
    #     f.close()
    # with open("beta_1_tree.txt", 'w+') as f:
    #     f.write(output1)
    #     f.close()



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
    parser.add_argument('--beta0', type=float, default=None)
    parser.add_argument('--beta1', type=float, default=None)
    parser.add_argument('--data-num', type=int, default=None)
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
        print("DDL: %d, coeff: %.0e" % (args.ddl, args.coeff))
        assert args.ddl is not None
        assert args.coeff is not None
        generate_best_beta_results(args.ddl, args.coeff)
    elif args.func == 'best':
        print("Looking for best FEC rate...")
        assert args.rtt is not None
        assert args.loss is not None
        assert args.ddl is not None
        assert args.group_delay is not None
        # return best betas with group-size provided
        best_beta, best_qoe, best_miss_rate, best_bw_loss, group_size = \
            get_best_beta_interface(args.rtt, args.ddl, args.loss, args.group_size, args.coeff, args.group_delay, data_num=args.data_num)
        print("best beta: %.10f, %.10f, group size: %d, ddl miss rate: %.10f, bw loss rate: %.10f, best qoe: %f" % (best_beta[0], best_beta[1], group_size, best_miss_rate, best_bw_loss, best_qoe))
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
        print("Calculating packet miss rate and bandwidth loss rate with Markov model...")
        assert args.rtt is not None
        assert args.loss is not None
        assert args.group_size is not None
        assert args.ddl is not None
        assert args.beta0 is not None
        assert args.beta1 is not None
        network_state = NetworkState()
        network_state.rtt = args.rtt
        network_state.loss_rate = args.loss
        network_state.group_delay = args.group_delay
        model = MarkovModel(network_state, args.group_size, args.ddl)
        miss_pbl, extra_pkt = model.build_2(args.beta0, args.beta1, verbose=True)
        print("At alpha: %.2f, beta_0: %.2f, beta_1: %.2f, miss_rate: %.10f, bw_loss_rate: %.10f, qoe: %f" % (args.loss, args.beta0, args.beta1, miss_pbl, extra_pkt / (args.group_size + extra_pkt), cal_qoe(miss_pbl, extra_pkt / (args.group_size + extra_pkt), args.coeff)))

