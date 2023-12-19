import os
import numpy as np
import argparse
import multiprocessing as mp
import re


def appProcess (fname):
    with open (fname, 'r') as f:
        lines = f.readlines ()
        lines = sorted ((line.split () for line in lines), key=lambda x: int (x[1]))
        lastFrameId = 0
        lastSecFrameId = 0
        missDdlFrame = 0
        missDdlTime = 0
        missRate = []
        missRateTime = []
        stallFreeTime = []
        frameDelay = []
        lastTime = 0
        lastAccTime = 0
        curStallFreeTime = 0
        for line in lines:
            if (int (line[8])):
                continue
            frameDelay.append (int (line[6]) - int (line[3]))
            curTime = int (int (line[3]) / 1000)
            curAccTime = int (line [3])
            curFrameId = int (line[1])
            if (curFrameId - lastFrameId > 1):
                missDdlFrame += curFrameId - lastFrameId - 1
                missDdlTime += curAccTime - lastAccTime - 100 / 6
            if curTime > lastTime:
                missRate.append (missDdlFrame / (curFrameId - lastSecFrameId))
                missRateTime.append (missDdlTime / 1000)
                if missDdlTime == 0:
                    curStallFreeTime += 1
                else:
                    stallFreeTime.append (curStallFreeTime)
                    curStallFreeTime = 0
                lastSecFrameId = curFrameId
                missDdlFrame = 0
                missDdlTime = 0
                lastTime = curTime
            lastFrameId = curFrameId
            lastAccTime = curAccTime
    if missRate == []:
        print (fname)
        missRate = [0]
    stallFreeTime.append (curStallFreeTime)
    return np.array (missRate), np.mean (frameDelay), np.array (missRateTime), np.array (stallFreeTime)


def fecProcess (fname):
    with open (fname, 'r') as f:
        lines = f.readlines ()
        dataPacket, otherPacket = 0, 0
        dataPackets, otherPackets = [], []
        lastTime = 0
        for line in lines:
            line = line.split ()
            curTime = int (int (line[0]) / 1000)
            if curTime > lastTime:
                dataPackets.append (dataPacket)
                otherPackets.append (otherPacket)
                otherPacket = 0
                dataPacket = 0
                lastTime = curTime
            isRtx = int (line[6])
            if (isRtx):
                otherPacket += int (line[10])
            else:
                dataPacket += int (line[10])
            otherPacket += int (line[12])
    assert (dataPackets != [])
    return np.array (otherPackets), np.array (dataPackets)


def calcVmaf (missRate, bwLoss, vmafModel):
    inputLen = 30
    calcRound = int (min (len (missRate) / inputLen, len (bwLoss) / inputLen))
    vmaf = []

    for i in range (calcRound):
        inputFeed = np.concatenate ((missRate[:inputLen], 1 - 1 / (1 + bwLoss[:inputLen]), np.ones (inputLen)))
        vmaf.append (vmafModel.predict (inputFeed.reshape (1, 3 * inputLen)))
        missRate = np.roll (missRate, -inputLen)
        bwLoss = np.roll (bwLoss, -inputLen)
    return vmaf


def multiAlgo (logdir, algo, args):
    traces = os.listdir (os.path.join (logdir, algo))
    traces = [trace for trace in traces if re.findall (args.tracefilter, trace)]
    missDdlAll = []
    missDdlTimeAll = []
    stallFreeTimeAll = []
    bwLossAll = []
    vmafSum = 0
    avgDelaySum = 0
    for trace in traces:
        missDdl, avgDelay, missDdlTime, stallFreeTime = appProcess (os.path.join (logdir, algo, trace, 'app.log'))
        otherPackets, dataPackets = fecProcess (os.path.join (logdir, algo, trace, 'fec.log'))
        missDdlAll.append (np.mean (missDdl))
        missDdlTimeAll.append (np.mean (missDdlTime))
        stallFreeTimeAll.append (np.mean (stallFreeTime))
        bwLossAll.append (np.sum (otherPackets) / np.sum (dataPackets))
        if args.vmaf:
            vmafSum += np.min (calcVmaf (missDdl, otherPackets / dataPackets, model))
        avgDelaySum += avgDelay
    return ("%s %.4f%% %.4f%% %.4f %.4f %.4f%% %.4f%% %.4f%% %d" % (algo, 
        np.mean (missDdlAll) * 100, 
        np.mean (bwLossAll) * 100, 
        vmafSum / len (traces),
        avgDelaySum / len (traces),
        np.percentile (missDdlAll, 95) * 100,
        np.percentile (bwLossAll, 95) * 100,
        np.mean (missDdlTimeAll) * 100,
        np.mean (stallFreeTimeAll)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser ()
    parser.add_argument ('--vmaf', action='store_true', help='calculate vmaf')
    parser.add_argument ("--tracefilter", type=str, default="(^.*$)")
    parser.add_argument ("--logdir", type=str, default='logs/')
    args = parser.parse_args ()

    if args.vmaf:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        # the VMAF prediction model used in Ke Chen, et al. in MMSys 2022.
        model = keras.Sequential ([
            layers.Dense(90, activation='relu', input_shape=[90]),
            layers.Dense(90, activation='relu'),
            layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam (0.001)
        model.compile(loss='mae',
            optimizer=optimizer,
            metrics=['mae', 'mse'])
        wights_file = 'Weights-494--5.61865.hdf5' # choose the best checkpoint 
        model.load_weights (wights_file)

    algos = [
        # 'dupackrtx', 'dupackbolot', 'dupackusf', 'dupackwebrtc', 'dupackawebrtc',
        # 'ptortx', 'ptobolot', 'ptousf', 'ptowebrtc', 'ptoawebrtc',
        # 'dupacklin0.50', 'dupacklin1.00', 'dupacklin2.00', 'dupacklin4.00', 
        # 'dupackhairpin1', 'dupackhairpin2', 'dupackhairpin4', 'dupackhairpin7',
        # 'dupackfixedrtx0.02', 'dupackfixedrtx0.05', 'dupackfixedrtx0.10', 'dupackfixedrtx0.20', 
        # 'dupackfixedrtx0.30', 'dupackfixedrtx0.50',
        # 'dupacktokenrtx',
        'dupackhairpinone1e+01', 'dupackhairpinone5e+00', 'dupackhairpinone1e+00', 'dupackhairpinone1', 'dupackhairpinone7',
        # 'dupackhairpin7'
    ]
    if not args.vmaf:
        pool = mp.Pool (int (mp.cpu_count () * 0.9))
        results = pool.starmap (multiAlgo, [(args.logdir, algo, args) for algo in algos], chunksize=1)
        pool.close ()
        pool.join ()
    else:
        results = [multiAlgo (args.logdir, algo, args) for algo in algos]
    for result in results:
        print (result)
