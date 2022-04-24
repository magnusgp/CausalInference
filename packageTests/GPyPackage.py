from GPy import Model, Param
import scipy
class Rosen(Model):
    def __init__(self, X, name='rosenbrock'):
        super(Rosen, self).__init__(name=name)
        self.X = Param("input", X)
        self.add_parameter(self.X)

    def log_likelihood(self):
        return -scipy.optimize.rosen(self.X)

    def parameters_changed(self):
        self.X.gradient = -scipy.optimize.rosen_der(self.X)

if __name__ == '__main__':
    m = Rosen(np.array([-1, -1]))
    print(m)