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

                data.append(data)
    return data


def plotGraph(data, title, filename ):
    
    ranks = [1,2,4,8,16,32,64]
    blockSizes = [16,32,64,128,256,512,1024]

    maxBandwidths = []

    for rank in ranks:
        rankSet = sorted([a for a in data if a['num_rank'] == rank])
        rankSet = sorted(rankSet, key = lambda i: i['output_file'] / i['time_secs'])
        maxBandwidth = rankSet[0]
        maxBandwidths.append

    ys = [i['output_file'] / i['time_secs'] for i in write_points]
        
    plt.title("{} bandwidth".format(title))
    plt.plot(range(len(ranks)), ys, 'o-', label = blockSize)


    labels = ["{}\n{}".format(a,b) for a,b in zip(ranks,blockSizes)]
    
    plt.xticks(ticks = ranks, labels = labels)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.savefig("{}.png".format(filename))
    plt.clf()
    


data = getData("StrongLow")
plotGraph(data, "First Strong Scaling Test", "StrongLow")

data = getData("StrongHigh")
plotGraph(data, "Second Strong Scaling Test", "StrongHigh")

data = getData("Weak")
plotGraph(data, "Weak Scaling Test", "Weak")
