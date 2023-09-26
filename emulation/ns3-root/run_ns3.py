import multiprocessing as mp
import subprocess as sp
import argparse
from tqdm import tqdm
import os

pbar = tqdm (smoothing=0, ncols=80)

def run_thread (conf):
    cmd = conf.split (',')[0]
    output = conf.split (',')[1].split ('\n')[0]
    if not os.path.exists (os.path.split(output)[0]):
        os.makedirs (os.path.split(output)[0])
    proc = sp.Popen ("./waf --run-no-build " + cmd + " > " + output + " 2>&1", shell=True)
    proc.wait ()


def pbar_update (*a):
    pbar.update ()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument ('--conf')
    parser.add_argument ('--worker', type=int, default=int (mp.cpu_count () * 0.9))
    args = parser.parse_args ()

    proc = sp.Popen ("./waf", shell=True)
    proc.wait ()

    with open (args.conf, 'r') as f:
        confs = f.readlines ()
    pbar.reset (total=len (confs))
    pool = mp.Pool (args.worker)
    for conf in confs:
        pool.apply_async (run_thread, (conf, ), callback=pbar_update)
    pool.close ()
    pool.join ()
    pbar.close ()