import os
import numpy as np
from writecdf import WriteCdf

def ReadLoss (fileName):
    loss = []
    with open (fileName, 'r') as f:
        for line in f.readlines ():
            loss.append (max (0, float (line.split ()[2])))
    return loss


if __name__ == '__main__':
    frameLoss = []
    sessionLoss = []
    for traceFile in os.listdir ('traces'):
        if 'eth' not in traceFile:
            continue
        loss = ReadLoss (os.path.join ('traces', traceFile))
        frameLoss.extend (loss)
        sessionLoss.append (np.mean (loss))
    
    WriteCdf ('frame-loss-eth.cdf', frameLoss)
    WriteCdf ('session-loss-eth.cdf', sessionLoss)
    
    frameLoss = []
    sessionLoss = []
    for traceFile in os.listdir ('traces'):
        if 'wifi' not in traceFile:
            continue
        loss = ReadLoss (os.path.join ('traces', traceFile))
        frameLoss.extend (loss)
        sessionLoss.append (np.mean (loss))
    
    WriteCdf ('frame-loss-wifi.cdf', frameLoss)
    WriteCdf ('session-loss-wifi.cdf', sessionLoss)