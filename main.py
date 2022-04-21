from sampleLoad import sampleLoad
from structEqModel import structEqModel

# Main script to run helper functions that will infer the graph

if __name__ == '__main__':
    model = structEqModel(n=1000)
    model.loadData(path='sample/data_435.csv')
    #print(model.data)
