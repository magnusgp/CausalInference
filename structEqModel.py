import numpy as np
import matplotlib.pyplot as plt

class structEqModel():
    def __init__(self, n, intervene = False):
        self.a = (np.random.exponential(1/7, n))
        self.b = (np.random.normal(1, 4, n))

        if intervene:
            self.c = 1
        else:
            self.c = np.where(np.random.multinomial(1, [1/6, 2/6, 3/6], n) == 1)[1]+1

        self.d = self.a + self.b + 30 * np.random.binomial(n, 2/5, None)
        self.e = self.b + 20 * self.c
        self.f = (2/3) * self.d + self.e

    def plotHist(self, dist=None, bins=20):
        import matplotlib.pyplot as plt
        if dist is not None:
            dists = ["a", "b", "c", "d", "e", "f"]
            distsModel = [self.a, self.b, self.c, self.d, self.e, self.f]

            if (len(dist) > 1):
                for i in range(len(dist)):
                    plt.suptitle('Model distribution ' + dist[i])
                    plt.hist(distsModel[dist.index(dist[i])], bins=bins, label=dists[dist.index(dist[i])])
                    plt.show()

            else:
                plt.hist(distsModel[dists.index(dist)], bins=bins)
                plt.show()
        else:
            print("Error: no distribution is given")

    def plotSamples(self, samples, node='all', pd = False):
        if pd and node != 'all':
            plt.hist(samples[node], bins=20)
            plt.show()

        else:
            all_nodes = ['A', 'B', 'C', 'D']
            if node !='all':
                nodeidx = all_nodes.index(node) + 1
                plt.hist(samples[nodeidx], bins = 5)
                plt.show()

            elif node=='all':
                nodes = all_nodes

            else:
                print("Error: node not found")

    def mixedVar(self, sigmax, sigmay, a, b, sigmaz=None, c=None):
        if sigmaz is None and c is None:
            return a ** 2 * sigmax ** 2 + b ** 2 * sigmay ** 2 + 2 * a * b * sigmax * sigmay
        else:
            return a ** 2 * sigmax ** 2 + b ** 2 * sigmay ** 2 + c ** 2 * sigmaz ** 2 + 2 * a * b * sigmax * sigmay * sigmaz

    def loadData(self, path, pd = False):
        from sampleLoad import sampleLoad, sampleLoadPd
        if pd:
            self.data = sampleLoadPd(path)
        else:
            self.data = sampleLoad(path)
        return self.data

if __name__ == "__main__":
    from tabulate import tabulate

    # set random seed for numpy
    np.random.seed(0)
    model = structEqModel(n=1000)

    print("Var[B] = ", model.mixedVar(2, (-1 / 2), 1 / 2, 1))
    print("\nVar[C] = ", model.mixedVar(2, 1, 1, 1))
    print("\nVar[D] = ", model.mixedVar(sigmax=model.mixedVar(2, (-1 / 2), 1 / 2, 1), sigmay=model.mixedVar(2, 1, 1, 1),
                                        a=2 / 3, b=-(1 / 2), sigmaz=1, c=1))

    data = [['A', round(np.mean(model.a), 3), round(np.var(model.a), 3)],
            ['B', round(np.mean(model.b), 3), round(np.var(model.b), 3)],
            ['C', round(np.mean(model.c), 3), round(np.var(model.c), 3)]]

    print("\n\nTable of expected values and variances:")
    print(tabulate(data, headers=['Variable', 'Expected value', 'Expected variance'], numalign="left"))

    # model.plotHist(dist=["a", "b", "c"])
    model.plotHist(dist=["d", "e", "f"])

    structEqModel(n=1000, intervene=True).plotHist(dist=["d", "e", "f"])

