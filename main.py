from sampleLoad import sampleLoad
from structEqModel import structEqModel
import numpy as np

# Main script to run helper functions that will infer the graph

if __name__ == '__main__':
    model = structEqModel(n=1000)
    samples = model.loadData(path='sample/data_435.csv', pd=True)
    # Plot pandas histograms for all nodes using pandas function
    model.plotSamples(samples, node='A')
    #print(model.data)
