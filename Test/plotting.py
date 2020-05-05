#!/bin/python3

import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

"""
Code to create a list of all device objects and all host objects
"""

def getData(directory):
    data = []
    for dir_path, _, file_names in os.walk(directory):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)

            if os.path.splitext(file_path)[1] == ".yaml":
                yaml_file = open(file_path)
                test = yaml.load(yaml_file, Loader=yaml.FullLoader)

                # Ignore empty files for now.
                # if test is not None:
                data.append(test)

    return data


def plotGraph(data, title, filename, xytexts=None ):
    print(filename)
    ranks = [1,2,4,8,16,32,64]
    blockSizes = [16,32,64,128,256,512,1024]

    maxBandwidths = []

    for rank in ranks:
        rankSet = [a for a in data if a['num_ranks'] == rank]
        # Sort ascending by bandwidth.
        rankSet = sorted(rankSet, key = lambda i: i['bytes_written'] / i['time_secs'])
        # print("rank = ", rank, end = "")
        # print("len = ", len(rankSet))
        maxBandwidth = rankSet[-1]
        maxBandwidths.append(maxBandwidth)

    ys = np.array([i['bytes_written'] / i['time_secs'] for i in maxBandwidths])
    times = [i['time_secs'] for i in maxBandwidths]
    print(times)
    labels = ["{}\n{}".format(a, b) for a, b in zip(ranks, blockSizes)]
    plt.figure(figsize= (8, 5))
    # plt.figure(figsize=(14,6))
    plt.title("{} Bandwidth".format(title))
    # print(ranks)
    ys = ys / (10**6)
    # print(ys)
    plt.plot(range(len(ranks)), ys, 'o-')
    # plt.ylim((-0.05*10**7,1.5*10**7))
    plt.xticks(ticks = range(len(ranks)), labels = labels)
    # plt.tight_layout()
    plt.margins(x=0.15, y = 0.15)

    for i,(x,y) in enumerate(zip(range(len(ranks)),ys)):

        if "Strong" in filename:
            label = "{:.2f}".format(y)
        else:
            label = "{:.2f}".format(y)

        xyt = None
        if xytexts is not None:
            xyt = xytexts[i]
        else:
            xyt=(20,10)

        plt.annotate(label, # this is the text
                    (x,y), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext = xyt, # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    plt.ylabel("Bandwidth: Mb/secs")
    xlabel = "\nConfiguration:         ranks\n \
                                thread count"
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig("John-{}.png".format(filename),dpi = 150)
    plt.clf()

data = getData("Output-John/StrongLow")
xyts = [(10,10),(12,5),(10,10),(10,10),(10,10),(10,10),(10,10)]
plotGraph(data, "First Strong Scaling Test", "StrongLow",xyts)

data = getData("Output-John/StrongHigh")
xyts = [(10,10),(20,10),(10,10),(10,10),(10,10),(10,10),(10,10)]
plotGraph(data, "Second Strong Scaling Test", "StrongHigh",xyts)

data = getData("Output-John/Weak")
xyts = [(20,5),(20,-7),(10,12),(20,10),(20,5),(20,-15),(10,7)]
plotGraph(data, "Weak Scaling Test", "Weak",xyts)
