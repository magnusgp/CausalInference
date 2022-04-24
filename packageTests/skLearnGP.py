from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

df = pd.read_csv('sample/data_569.csv', index_col = 0)
x = df['A']
X = x.reshape(-1, 1)
print(X.shape)

gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
gp.fit(X, y)
GaussianProcessRegressor(alpha=1e-10, copy_X_train=True,
kernel=1**2 + Matern(length_scale=2, nu=1.5) + WhiteKernel(noise_level=1),
n_restarts_optimizer=0, normalize_y=False,
optimizer='fmin_l_bfgs_b', random_state=None)