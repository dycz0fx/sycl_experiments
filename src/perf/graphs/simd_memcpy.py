import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np

PUSH01 = 0
PULL01 = 1
PUSH02 = 2
PULL02 = 3
SAME = 4

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


with open('simd.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    rawdata = [row for row in reader]

    
data = []
for r in rawdata:
    print(r)
    data.append(cleandata(r))
#print(data)

def sgsizevswgsize(mode, size, title, fname):
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes()
    plt.xlabel('wgsize')
    plt.ylabel('MB/s')
    plt.xscale('log')
    plt.yscale('log')
    for sgsize in [16,32]:
        if sgsize == 16:
            wgsizes = [16,32,64,128,256,512,1024]
        if sgsize == 32:
            wgsizes = [32,64,128,256,512,1024]
        print('sgsize', sgsize)
        x = [r['wgsize'] for r in data if r['mode'] == mode and r['size'] == size and r['sgsize']==sgsize]
        y = [r['bandwidth'] for r in data if r['mode'] == mode and r['size'] == size and r['sgsize']==sgsize]
        print('plot', x, y)
        plt.plot(x, y, label=str(sgsize))
    plt.title(title)
    leg = plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(fname, format='pdf')
    plt.show()

    
modetext = ['push_cross_tile','pull_cross_tile','push_cross_device','pull_cross_device','same_tile', "pull_host", "push_host"]
def sizevswgsize(mode, title, fname):
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes()
    plt.xlabel('size')
    plt.ylabel('MB/s')
    plt.xscale('log')
    plt.yscale('log')
    for sgsize in [32]:
        wgsizes = [32,64,128,256,512,1024]
        for wgsize in wgsizes:
            x = [r['size'] for r in data if r['mode'] == mode and r['wgsize'] == wgsize and r['sgsize'] == sgsize]
            y = [r['bandwidth'] for r in data if r['mode'] == mode and r['wgsize'] == wgsize and r['sgsize'] == sgsize]
            print("plot", x, y)
            plt.plot(x, y, label=str(str(wgsize)))
    plt.title(title)
    leg = plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(fname, format='pdf')
    plt.show()

zerotime = 0.140
zerocount = 8192.0
zeroeach = zerotime/zerocount

def plotall(cmd, mode, title, fname):
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes()
    plt.xlabel('size')
    plt.ylabel('MB/s')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1.0,1000000.0)
    plt.xlim(1.0,5000000000.0)
    matchdata = [r for r in data if r['cmd'] == cmd and r['mode'] == mode]
    #print(matchdata)
    # find wg_sizes
    wgsizes = list(set([r['wgsize'] for r in matchdata]))
    wgsizes.sort()
    print(wgsizes)
    for wgsize in wgsizes:
        x = [r['size'] for r in matchdata if r['wgsize'] == wgsize]
        y = [(r['size']/1000000.0)/((r['duration']/r['count']) - zeroeach)  for r in matchdata if r['wgsize'] == wgsize]
        #y = [r['bandwidth'] for r in matchdata if r['wgsize'] == wgsize]
        print("plot", x, y)
        plt.plot(x, y, label=str(str(wgsize)))
    for wgsize in [1024]:
        x = [r['size'] for r in data if r['cmd'] == 2 and r['mode'] == mode and r['wgsize'] == wgsize]
        y = [(r['size']/1000000.0)/((r['duration']/r['count']) - zerotime) for r in data if r['cmd'] == 2 and  r['mode'] == mode and r['wgsize'] == wgsize]
        #y = [r['bandwidth'] for r in data if r['cmd'] == 2 and r['mode'] == mode and r['wgsize'] == wgsize]
        print("plot", x, y)
        plt.plot(x, y, label="range")
    plt.title(title)
    leg = plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(fname, format='pdf')
    plt.show()
    

    
cmdtext = ["nd_range", "work_group", "range"]

for cmd in range(2):
    for mode in range(7):
        plotall(cmd, mode, cmdtext[cmd]+"_"+modetext[mode], cmdtext[cmd]+ "_" +modetext[mode]+".pdf")
