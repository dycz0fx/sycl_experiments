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


with open('new.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    data = [cleandata(row) for row in reader]



def sizevsthreads(mode, title, fname):
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes()
    plt.xlabel('Threads')
    plt.ylabel('MB/s')
    ax.set_xscale('function', functions=(forward,inverse))
    ax.set_xticks([1<<n for n in range(1,11)])
    plt.yscale('log')
    for size in [1<<x for x in range(1,27)][::-1]:
        dataset = [r for r in data if r['mode'] == mode and r['size'] == size]
        x = [r['threads'] for r in dataset]
        y = [r['bandwidth'] for r in dataset]
        plt.plot(x, y, label=str(size))
        print (size)
    plt.title(title)
    leg = plt.legend(loc='lower right',ncol=2)
    plt.savefig(fname, format='pdf')
    plt.show()

sizevsthreads(PUSH, "Xe Push, size vs threads",'push.pdf')
sizevsthreads(PULL, "Xe Pull, size vs threads", 'pull.pdf')
sizevsthreads(SAME, "Same GPU, size vs threads", 'same.pdf')
