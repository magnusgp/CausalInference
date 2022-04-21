import numpy as np

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

