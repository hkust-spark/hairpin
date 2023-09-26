'''
Model FEC and packet retransmission together
using Markov chain
'''
import math
import time
from scipy import special
from multiprocessing import Pool
import argparse
import numpy as np
import logging

# calculate transition probability from S(r0, n0) to S(r1, n1)
def cal_trans_pbl(args, alpha, beta_list, r0, n0, r1, n1):
    try:
        assert r0 < r1
        assert r0 >= 0
        assert n0 >= n1
        assert n0 <= args.N
        assert n1 >= 0
    except AssertionError as e:
        logging.error("fec_w_rtx: Assertion Error in cal_trans_pbl(%d, %d, %d, %d)" % (r0, n0, r1, n1))
    N0 = args.N
    beta = beta_list[r0]
    fec_count = int(math.floor(n0 * beta))
    # S(r0, n0) -> S(r1, 0)
    if n1 == 0:
        p01 = 0
        for i in range(0, fec_count + 1):
            p01 = p01 + \
                special.comb(fec_count + n0, i, exact=True) * alpha ** i * (1 - alpha) ** (fec_count + n0 - i)
        return p01

    # S(r0, n0) -> S(r1, n1)
    if n1 != 0:
        p01 = 0
        min_loss_count = int(max(fec_count + 1, n1))
        assert min_loss_count <= n1 + fec_count
        for i in range(min_loss_count, n1 + fec_count + 1):
            assert n1 <= i
            p01 = p01 + \
                special.comb(fec_count + n0, i, exact=True) * alpha ** i * (1 - alpha) ** (fec_count + n0 - i) * \
                special.comb(fec_count, i - n1, exact=True) * special.comb(n0, n1, exact=True) / special.comb(fec_count + n0, i, exact=True)
        return p01

# calculate probability of state S(r, n)
def cal_state(args, alpha, beta_list, matrix, r, n):
    # layer 0: S(0, N0)
    if r == 0:
        if n == args.N:
            return 1
        else:
            return 0
    try:
        assert n <= args.N
        assert r >= 1
        assert r <= args.layer - 1
    except AssertionError as e:
        logging.error("fec_w_rtx: Assertion Error in cal_state(%d, %d)" % (r, n))
    # layer 1 ~ args.layer - 1
    pbl = 0
    min_start_n = int(max(1, n))
    for i in range(min_start_n, args.N + 1):
        trans_pbl = cal_trans_pbl(args, alpha, beta_list, r - 1, i, r, n)
        if trans_pbl > 1 and trans_pbl - 1 <= 1e-10:
            trans_pbl = 1
        try:
            assert trans_pbl <= 1
        except AssertionError as e:
            logging.error("fec_w_rtx: Transition pbl from S(%d,%d) to S(%d,%d) is %3.30f " % (r-1, i, r, n, trans_pbl))
        pbl = pbl + get_state(args, matrix, r - 1, i) * trans_pbl
    return pbl

# return stored probablity of state S(r, n)
def get_state(args, matrix, r, n):
    try:
        assert n <= args.N
        assert r <= args.layer - 1
    except AssertionError as e:
        logging.error("fec_w_rtx: Assertion Error in get_state(%d, %d)" % (r, n))
    result = matrix[r][n]
    assert result is not None
    return result

# calculate the probability of every state
def build_markov(args, alpha, beta_list, check=False):
    # init Markov state matrix
    matrix = [None] * args.layer
    for i in range(0, args.layer):
        matrix[i] = [None] * (args.N + 1)

    # calculation
    logging.info("fec_w_rtx: Building markov matrix with alpha: %3.3f" % (alpha, ))
    for i in range(0, args.layer):
        for j in range(0, args.N + 1):
            matrix[i][j] = \
                cal_state(args, alpha, beta_list, matrix, i, j)
    if check:
        check_markov(args, matrix)
    # print("Exit at layers: " + str([col[0] for col in matrix][1:]))
    return matrix_feature(args, beta_list, matrix)

# return features of the matrix
def matrix_feature(args, beta_list, matrix):
    # calculate expected num of pkts missing ddl
    expected_miss = 0
    for i in range(1, args.N + 1):
        expected_miss = expected_miss + matrix[args.layer - 1][i] * i
    logging.info("fec_w_rtx: The expected num of pkts missing ddl is %3.15f." % expected_miss)

    # calculate expected packets transmitted
    extra_pkt = 0
    for i in range(1, args.layer):
        for j in range(1, args.N + 1):
            extra_pkt = extra_pkt + (j + math.floor(j * beta_list[i])) * matrix[i][j]
    # layer 0
    extra_pkt = extra_pkt + math.floor(args.N * beta_list[0])
    return expected_miss, extra_pkt

# for test purpose
# check if the probability calculated is correct
def check_markov(args, matrix):
    # print the matrix
    for i in range(0, args.N + 1):
        out_str = ""
        for j in range(0, args.layer):
            if j == 0:
                if i == args.N:
                    out_str = out_str + ("S(%d, %d) = %3.15f\t" % (j, i, matrix[0][0]))
                else:
                    out_str = out_str + ("\t" * 4)
            elif j != 0:
                out_str = out_str + ("S(%d, %d) = %3.15f\t" % (j, i, matrix[j][i]))
        logging.info("fec_w_rtx: " + out_str)


    # check if the sum of the second column is 1
    sum = 0
    for i in range(0, args.N + 1):
        sum = sum + matrix[1][i]
    logging.info("fec_w_rtx: The sum of r == 1 is %3.15f" % sum)

    missing_sum = 0
    not_missing_sum = 0
    # print the sum of "missing ddl" states
    for i in range(1, args.N + 1):
        missing_sum = missing_sum + matrix[args.layer - 1][i]
    logging.info("fec_w_rtx: The sum of \"missing ddl\" states is %3.15f." % missing_sum)
    # print the sum of "not missing ddl" states
    for i in range(1, args.layer):
        not_missing_sum = not_missing_sum + matrix[i][0]
    logging.info("fec_w_rtx: The sum of \"not missing ddl\" states is %3.15f." % not_missing_sum)
    # check if the sum of the exiting states is 1
    logging.info("fec_w_rtx: The sum of all exiting states is %3.15f." % (missing_sum + not_missing_sum))

    for i in range(1, args.layer):
        c_sum = 0
        for j in range(1, i):
            c_sum = c_sum + matrix[j][0]
        for j in range(0, args.N + 1):
            c_sum = c_sum + matrix[i][j]
        logging.info("fec_w_rtx: The sum of column %d is %3.15f" % (i, c_sum))

    # calculate expected num of pkts missing ddl
    e_miss = 0
    for i in range(1, args.N + 1):
        e_miss = e_miss + matrix[args.layer - 1][i] * i
    logging.info("fec_w_rtx: The expected num of pkts missing ddl is %3.15f." % e_miss)


def qoe(args, expected_miss, extra_pkt, c):
    return - expected_miss / args.N + c * args.N / (args.N + extra_pkt)


def build_markov_process(params):
    args, alpha, beta_0, b1_list = params
    result_list = []
    for beta_1 in b1_list:
        beta_list = [beta_0] + [beta_1] * (args.layer - 1)
        expected_miss, extra_pkt = build_markov(args, alpha, beta_list)
        result_list.append((beta_1, expected_miss, extra_pkt))
    return (beta_0, result_list)


def find_best_beta(args):
    if args.verbose:
        for f in args.pkl_fnames:
            if os.path.exists(f):
                os.remove(f)

    if args.loss:
        a_list = [args.loss]
    else:
        a_list = np.arange(0.01, 0.51, 0.01)

    b0_list = np.arange(0, 1.02, 0.02)
    b1_list = np.arange(0, 5.1, 1)
    if args.coeff:
        c_list = [args.coeff]
    else:
        c_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    best_beta_dict = {}
    for c in c_list:
        best_beta_dict[c] = []
    for alpha in a_list:
        alpha_start_time = time.time()
        if args.verbose:
            logging.info("fec_w_rtx: alpha: %f, time: %.1f" % (alpha, time.time()))

        # test: calculate baseline
        bl_exp_miss, bl_extra_pkt = build_markov(args, alpha, [0] * args.N)
        if args.verbose:
            with open(args.pkl_fnames[2], 'a') as f:
                f.write("Baseline:\t alpha: %.2f, beta_0: %.2f, beta_1: %.2f, exp_miss: %.20f, extra_pkt: %.20f\n" % (alpha, 0, 0, bl_exp_miss, bl_extra_pkt))
                f.flush()
                f.close()

        pool = Pool()
        logging.info("fec_w_rtx: Building markov matrix with alpha: %.5f" % (alpha))
        # markov_feature_list = [(expected_miss, extra_pkt), ...]
        markov_feature_list = pool.map(build_markov_process, [(args, alpha, beta_0, b1_list) for beta_0 in b0_list])
        pool.close()
        pool.join()

        # test: comparison with baseline
        for (beta_0, result_list) in markov_feature_list:
            for (beta_1, expected_miss, extra_pkt) in result_list:
                if expected_miss < bl_exp_miss and extra_pkt < bl_extra_pkt:
                    if args.verbose:
                        with open(args.pkl_fnames[2], 'a') as f:
                            f.write("\t\t\t alpha: %.2f, beta_0: %.2f, beta_1: %.2f, exp_miss: %.20f, extra_pkt: %.20f\n" % (alpha, beta_0, beta_1, expected_miss, bl_exp_miss))
                            f.flush()
                            f.close()

        # now we have results for every (beta_0, beta_1) pair given alpha
        # store it
        if args.verbose:
            with open(args.pkl_fnames[1], 'ab') as f:
                pickle.dump((alpha, markov_feature_list), f)
                f.flush()
                f.close()

        # calculate best beta pair at given alpha
        for c in c_list:
            # calculate best beta pair at given qoe model (c)
            best_qoe = -1e10
            best_beta = None
            for (beta_0, results) in markov_feature_list:
                for (beta_1, expected_miss, extra_pkt) in results:
                    # given alpha, for every (beta_0, beta_1)
                    current_qoe = qoe(args, expected_miss, extra_pkt, c)
                    if current_qoe > best_qoe:
                        best_qoe = current_qoe
                        best_beta = (beta_0, beta_1)
            assert best_beta is not None
            best_beta_dict[c].append(best_beta)
        if args.verbose:
            logging.info("fec_w_rtx: alpha: %3.3f ends, time consume: %.1f s" % (alpha, time.time() - alpha_start_time))
        del markov_feature_list

    # store best_beta_dict
    if args.verbose:
        with open(args.pkl_fnames[0], 'wb') as f:
            pickle.dump(a_list, f)
            pickle.dump(best_beta_dict, f)
            f.flush()
            f.close()
    return a_list, best_beta_dict

def load_best_beta(filename):
    a_list = None
    best_beta_dict = None
    with open(filename, 'rb') as f:
        a_list = pickle.load(f)
        best_beta_dict = pickle.load(f)
        f.close()
    return a_list, best_beta_dict

def plot_best_beta(args, a_list, best_beta_dict):
    fig, axes = plt.subplots(1, 2)
    for c, best_beta_list in best_beta_dict.items():
        axes[0].plot(a_list, [b[0] for b in best_beta_list], label=str(c))
        axes[1].plot(a_list, [b[1] for b in best_beta_list], label=str(c))

    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$\beta_0$")
    axes[0].set_xlim(0, a_list[-1])
    axes[0].set_ylim(0, 1)
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$\beta_1$")
    axes[1].set_xlim(0, a_list[-1])
    axes[1].set_ylim(0, 5)
    plt.legend(title=r"$c$")
    plt.savefig('fec_w_rtx_%d_%dms_%dms_%dMbps.pdf' % (args.fec_num, args.rtt, args.deadline, args.bandwidth))
    plt.show()

def martix_feature_to_text(binary_filename):
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

def extend_args(args):
    args.N = args.fec_num
    # args.N = int(math.floor(args.bandwidth / 8 * 1e6 * args.delta_t / 1000 / 1500))
    # args.N = 50
    args.epsilon = int(math.floor((args.deadline - args.rtt) / args.rtt)) + 1
    # num of Markov chain layers
    args.layer = int(args.epsilon) + 1
    args.pkl_fnames = ['best_beta_%d_%dms_%dms_%dMbps.pkl' % (args.fec_num, args.rtt, args.deadline, args.bandwidth),
                       'matrix_feature_%d_%dms_%dms_%dMbps.pkl' % (args.fec_num, args.rtt, args.deadline, args.bandwidth),
                       'matrix_above_baseline_%d_%dms_%dms_%dMbps.txt' % (args.fec_num, args.rtt, args.deadline, args.bandwidth)]


# interface for python simulator to call
# given rtt, bandwidth, deadline, loss rate, fec group size and QoE coeff
# return (beta_0, beta_1) that makes the highest QoE
def run_model(rtt: int, bw: int, ddl: int, loss: float, fec_num: int=10, coeff: float=1e-4) -> tuple:
    print("Running model... loss: %f" % loss)
    if loss == 0:
        # loss rate = 0, no FEC needed
        return (0, 0)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.rtt = rtt
    args.deadline = ddl
    args.bandwidth = bw
    args.fec_num = fec_num
    args.loss = loss
    args.coeff = coeff
    args.verbose = False
    extend_args(args)
    a_list, best_beta_dict = find_best_beta(args)
    logging.info("fec_w_rtx.run_model: Best b0: %.5f, b1:%.5f at alpha: %.5f" % (best_beta_dict[args.coeff][0][0], best_beta_dict[args.coeff][0][1], loss))
    return best_beta_dict[coeff][0]

if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtt', type=int, default=20)
    parser.add_argument('--deadline', type=int, default=80)
    parser.add_argument('--bandwidth', type=int, default=30)
    parser.add_argument('--fec-num', type=int, default=10)
    parser.add_argument('--loss', type=float, default=None)
    parser.add_argument('--coeff', type=float, default=1e-4)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--beta0', type=float, default=None)
    parser.add_argument('--beta1', type=float, default=None)
    parser.add_argument('--func', type=str, default='best_beta')
    args = parser.parse_args()
    extend_args(args)

    logging.basicConfig(filename='fec_model.log', format='%(asctime)s %(message)s', level=logging.DEBUG)


    if args.verbose:
        import pickle
        import json
        import matplotlib.pyplot as plt
        import os

    if args.func == 'best_beta':
        a_list, best_beta_dict = find_best_beta(args)
        # a_list, best_beta_dict = load_best_beta(args.pkl_fnames[0])
        if args.verbose:
            plot_best_beta(args, a_list, best_beta_dict)
            martix_feature_to_text(args.pkl_fnames[1])
        elif args.coeff:
            print(best_beta_dict[args.coeff][0])
            logging.info(best_beta_dict[args.coeff][0])
    elif args.func == 'model':
        assert args.loss is not None
        assert args.beta0 is not None
        assert args.beta1 is not None
        expected_miss, extra_pkt = build_markov(args, args.loss, [args.beta0] + [args.beta1] * args.layer)
        print("At alpha: %.2f, beta_0: %.2f, beta_1: %.2f, pkt_miss_rate: %f, bw_loss_rate: %f, qoe: %f" % (args.loss, args.beta0, args.beta1, expected_miss / args.N, args.N / (args.N + extra_pkt), qoe(args, expected_miss, extra_pkt, args.coeff)))

