import argparse

def WriteCdf (fileName, data):
    data.sort ()
    curIndex = 0
    if (len (data) > 10000):
        stepSize = len (data) / 10000
    else:
        stepSize = 1
    with open (fileName, 'w') as f:
        lastData = -1
        while curIndex < len (data):
            if data[int (curIndex)] != lastData:
                f.write ("%.4f\t%.4f\n" % (data[int (curIndex)], curIndex / len (data)))
                lastData = data[int (curIndex)]
            curIndex += stepSize
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser ()
    parser.add_argument ('--input', type=str, required=True)
    parser.add_argument ('--output', type=str, required=True)
    args = parser.parse_args ()
    data = []
    with open (args.input, 'r') as f:
        lines = f.readlines ()
        for line in lines:
            try:
                data.append (float (line))
            except ValueError:
                pass
    WriteCdf (args.output, data)
