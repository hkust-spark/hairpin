import argparse
import pickle
import os
import time
# from sklearn import tree
# from sklearn_porter import Porter
# import graphviz 
import itertools
import numpy as np
from array import array


def load_results(coeff):
    beta_result = "model_result_%.0e.pkl" % (coeff)
    block_result = "block_result_%.0e.pkl" % (coeff)

    with open(beta_result, 'rb') as f:
        beta_list = pickle.load(f)
    with open(block_result, 'rb') as f:
        block_list = pickle.load(f)

    return beta_list, block_list

def generate_decision_tree(result_list):
    # reorder result_list
    group_result_dict : dict[int:list] = {}
    for [[rtt, loss, group_delay, group_size], [count_0, beta_1]] in result_list:
        if group_result_dict.get(group_size) is None:
            group_result_dict[group_size] = []
        group_result_dict[group_size].append([[rtt, loss, group_delay], [count_0, beta_1]])

    print(group_result_dict.keys())

    tree_dict : dict[int:tuple[tree.DecisionTreeClassifier]] = {}
    print("Start training at %d" % time.time())
    # train two decision trees per group_size
    for group_size, results in group_result_dict.items():
        print("Start training for  group size %d, result list length: %d" % (group_size, len(results)))
        tree_dict[group_size] = \
            (tree.DecisionTreeClassifier(), tree.DecisionTreeClassifier())
        tree_dict[group_size][0].fit([network for (network, result) in results], [result[0] for (network, result) in results])
        tree_dict[group_size][1].fit([network for (network, result) in results], [result[1] for (network, result) in results])
        print("Tree depth: %d, %d, leaf nodes: %d, %d" % (tree_dict[group_size][0].tree_.max_depth, tree_dict[group_size][1].tree_.max_depth, tree_dict[group_size][0].tree_.n_leaves, tree_dict[group_size][1].tree_.n_leaves))
        for i in (0, 1):
            dot_data = tree.export_graphviz(tree_dict[group_size][i], out_file=None)
            graph = graphviz.Source(dot_data)
            graph.render("tree_%d_%d" % (group_size, i))


def print_result_list(result_list):
    rtt_set = sorted(set([rtt for [[rtt, loss, group_delay, group_size], [count_0, beta_1]] in result_list]))
    print(("RTT: %d. " + str(rtt_set)) % len(rtt_set))
    loss_set = sorted(set([loss for [[rtt, loss, group_delay, group_size], [count_0, beta_1]] in result_list]))
    print(("Loss Rate: %d. " + str(loss_set)) % len(loss_set))
    group_delay_set = sorted(set([group_delay for [[rtt, loss, group_delay, group_size], [count_0, beta_1]] in result_list]))
    print(("Group Delay: %d. " + str(group_delay_set)) % len(group_delay_set))
    group_size_set = sorted(set([group_size for [[rtt, loss, group_delay, group_size], [count_0, beta_1]] in result_list]))
    print(("Group Size: %d. " + str(group_size_set)) % len(group_size_set))
    count_0_set = sorted(set([count_0 for [[rtt, loss, group_delay, group_size], [count_0, beta_1]] in result_list]))
    print(("Count 0: %d. " + str(count_0_set)) % len(count_0_set))
    beta_1_set = sorted(set([beta_1 for [[rtt, loss, group_delay, group_size], [count_0, beta_1]] in result_list]))
    print(("Beta 1: %d. " + str(beta_1_set)) % len(beta_1_set))
    return (rtt_set, loss_set, group_delay_set, group_size_set), (count_0_set, beta_1_set)


# result_set = (rtt_set, loss_set, group_delay_set, group_size_set), (count_0_set, beta_1_set)
# result_list = [[[rtt, loss, group_delay, group_size], [count_0, beta_1]], ...]
def generate_array(result_set, beta_dict, block_dict, coeff):
    # generate calculation equation
    loss_set, frame_size_set, layer_set, pkt_set, ddl_set, rtt_set, rdisp_set = result_set
    assert len(loss_set) >= 2
    loss_count = len(loss_set)
    loss_itvl = loss_set[1] - loss_set[0]
    assert len(frame_size_set) >= 2
    frame_size_count = len(frame_size_set)
    frame_size_itvl = frame_size_set[1] - frame_size_set[0]
    assert len(layer_set) >= 2
    layer_count = len(layer_set)
    layer_itvl = layer_set[1] - layer_set[0]
    assert len(pkt_set) >= 2
    pkt_count = len(pkt_set)
    pkt_itvl = pkt_set[1] - pkt_set[0]
    assert len(ddl_set) >= 2
    ddl_count = len(ddl_set)
    ddl_itvl = ddl_set[1] - ddl_set[0]
    assert len(rtt_set) >= 2
    rtt_count = len(rtt_set)
    rtt_itvl = rtt_set[1] - rtt_set[0]
    assert len(rdisp_set) >= 2
    rdisp_count = len(rdisp_set)
    rdisp_itvl = rdisp_set[1] - rdisp_set[0]

    # beta packet array
    beta_array_length = loss_count * frame_size_count * layer_count * pkt_count
    beta_array = [255] * beta_array_length
    for (loss, frame_size, layer, pkt) in itertools.product(loss_set, frame_size_set, layer_set, pkt_set):
        index = round((loss - loss_set[0]) / loss_itvl) * (frame_size_count * layer_count * pkt_count) \
            + round((frame_size - frame_size_set[0]) / frame_size_itvl) * (layer_count * pkt_count) \
            + round((layer - layer_set[0]) / layer_itvl) * (pkt_count) \
            + round((pkt - pkt_set[0]) / pkt_itvl)
        try:
            beta_array[index] = beta_dict[loss][frame_size][layer][pkt]['beta_pkt']
            beta_array[index] = beta_array[index] if beta_array[index] < 255 else 255
        except KeyError:
            continue

    # block size array
    block_array_length = loss_count * frame_size_count * ddl_count * rtt_count * rdisp_count
    block_array = [255] * block_array_length
    for (loss, frame_size, ddl, rtt, rdisp) in itertools.product(loss_set, frame_size_set, ddl_set, rtt_set, rdisp_set):
        index = round((loss - loss_set[0]) / loss_itvl) * (frame_size_count * ddl_count * rtt_count * rdisp_count) \
            + round((frame_size - frame_size_set[0]) / frame_size_itvl) * (ddl_count * rtt_count * rdisp_count) \
            + round((ddl - ddl_set[0]) / ddl_itvl) * (rtt_count * rdisp_count) \
            + round((rtt - rtt_set[0]) / rtt_itvl) * (rdisp_count) \
            + round((rdisp - rdisp_set[0]) / rdisp_itvl)
        try:
            block_array[index] = block_dict[loss][frame_size][ddl][rtt][rdisp]
        except KeyError:
            continue

    cap = 0
    rtx = 1
    beta_array_output = bytearray(beta_array)
    block_array_output = bytearray(block_array)
    with open("beta-array-rtx%d-cap%d-coeff%.0e.bin" % (rtx, cap, coeff), 'wb') as f:
        f.write(beta_array_output)
    with open("block-array-rtx%d-cap%d-coeff%.0e.bin" % (rtx, cap, coeff), 'wb') as f:
        f.write(block_array_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coeff', type=float, default=1e-4)
    args = parser.parse_args()

    print("Load result list at %d, coeff: %.0e" % (time.time(), args.coeff))
    # result_list = [[[rtt, loss, group_delay, group_size], [count_0, beta_1]], ...]
    beta_dict, block_dict = load_results(args.coeff)

    # TODO: move the parameters in fec_w_rtx and merge_result together to, e.g., a local file.
    loss_rate_start = 0.0
    loss_rate_end = 0.50
    loss_rate_itvl = 0.01
    loss_set = np.linspace(loss_rate_start, loss_rate_end, int(round((loss_rate_end - loss_rate_start) / loss_rate_itvl)) + 1).tolist()

    frame_size_start = 5
    frame_size_end = 55
    frame_size_itvl = 5
    frame_size_set = list(range(frame_size_start, frame_size_end + 1, frame_size_itvl))
    
    layer_set = list(range(1, 16))
    pkt_set = list(range(1, frame_size_end + 1))

    # ddl from 20 to 140
    ddl_start = 20
    ddl_end = 140
    ddl_step = 20
    ddl_set = list(range(ddl_start, ddl_end + 1, ddl_step))

    # rtt from 6 to 80
    rtt_start = 10
    rtt_end = 80
    rtt_step = 2
    rtt_set = list(range(rtt_start, rtt_end + 1, rtt_step))

    # dispersion rate from 0ms/pkt to 1ms/pkt
    rdisp_start = 0
    rdisp_end = 1
    rdisp_step = 0.02
    rdisp_set = np.linspace(rdisp_start, rdisp_end, int(round((rdisp_end - rdisp_start) / rdisp_step)) + 1).tolist()

    result_set = loss_set, frame_size_set, layer_set, pkt_set, ddl_set, rtt_set, rdisp_set

    # generate a big array
    generate_array(result_set, beta_dict, block_dict, args.coeff)
