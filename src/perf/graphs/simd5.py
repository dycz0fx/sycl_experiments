import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np

PUSH = 0
PULL = 1
SAME = 2

def cleandata(d):
    for k in d.keys():
        try:
            v = int(d[k])
            d[k] = v
        except ValueError:
            try:
                v = float(d[k])
                d[k] = v
            except ValueError:
                continue
    return(d)

# magic for powers of 2 on x axis

def forward(x):
    return np.log2(x)


def inverse(x):
    return 2**x


with open('simd5.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    rawdata = [row for row in reader]

    
data = [cleandata(r) for r in rawdata]


    
modetext = ['push','pull','push2','pull2','same']
def sizevsthreads(mode, title, fname):
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes()
    plt.xlabel('size')
    plt.ylabel('MB/s')
    plt.xscale('log')
    plt.yscale('log')
    for mode in [0,1,4]:
        x = [r['size'] for r in data if r['mode'] == mode and r['size'] != 1]
        zerotime = 0
        for r in data:
            if r['mode'] == mode and r['size'] == 1:
                zerocount = r['count']
                zeroduration = r['duration']
                zerotime = zeroduration / zerocount
        y = [(r['count'] * r['size'])/(1000000.0 * (r['duration'] - ((zeroduration * r['count'])/zerocount))) for r in data if r['mode'] == mode and r['size'] != 1]
        
        plt.plot(x, y, label=str(modetext[mode]))
    plt.title(title)
    leg = plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(fname, format='pdf')
    plt.show()

sizevsthreads(PUSH, "PVC SIMD memcpy",'simd5.pdf')
