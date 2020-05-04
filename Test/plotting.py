#!/bin/python3

import os
import yaml
import matplotlib.pyplot as plt

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
                if test is not None:
                    data.append(test)

    return data


def plotGraph(data, title, filename ):
    ranks = [1,2,4,8,16,32,64]
    blockSizes = [16,32,64,128,256,512,1024]

    maxBandwidths = []

    for rank in ranks:
        rankSet = [a for a in data if a['num_ranks'] == rank]
        # Sort ascending by bandwidth.
        rankSet = sorted(rankSet, key = lambda i: i['bytes_written'] / i['time_secs'])
        maxBandwidth = rankSet[-1]
        maxBandwidths.append(maxBandwidth)

    ys = [i['bytes_written'] / i['time_secs'] for i in maxBandwidths]

    labels = ["{}\n{}".format(a, b) for a, b in zip(ranks, blockSizes)]

    plt.title("{} bandwidth".format(title))
    plt.plot(range(len(ranks)), ys, 'o-')

    plt.xticks(ticks = ranks, labels = labels)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.savefig("{}.png".format(filename))
    plt.clf()

data = getData("Output/StrongLow")
plotGraph(data, "First Strong Scaling Test", "StrongLow")

data = getData("Output/StrongHigh")
plotGraph(data, "Second Strong Scaling Test", "StrongHigh")

data = getData("Output/Weak")
plotGraph(data, "Weak Scaling Test", "Weak")
