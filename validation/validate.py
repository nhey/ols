import statsmodels.api as sm
import numpy as np
from ols_test_pyopencl import ols_test_pyopencl

duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
Y = duncan_prestige.data['income']
X = duncan_prestige.data['education']
X = sm.add_constant(X)

# Note: the copy is really important here.
# GPU results are garbage without it.
X, y = np.array(X, dtype=np.float64).copy(), np.array(Y, dtype=np.float64).copy()
model = sm.OLS(y,X)
results = model.fit(method="qr")

ols_fut = ols_test_pyopencl()

Q, R = np.linalg.qr(X, mode="complete")
manual_cov_params = np.linalg.inv(np.dot(R.T, R))

ocl_params, ocl_cov_params = ols_fut.fit(1, X, y)

print("\n Covariance of parameters")
print("manual:", manual_cov_params)
print("sm.OLS:", results.normalized_cov_params)
print("opencl:", ocl_cov_params.get())

print("\n Parameters")
print("sm.OLS:", results.params)
print("opencl:", ocl_params)


def validate(num_tests, bsz):
  ok = True
  for i in range(num_tests):
    X = np.random.rand(100,6).astype(np.float64)
    y = np.random.rand(100,1).astype(np.float64)
    sm_results = sm.OLS(y,X).fit(method="qr")
    sm_params = sm_results.params
    sm_cov_params = sm_results.normalized_cov_params

    ocl_params, ocl_cov_params = ols_fut.fit(bsz, X, y)

    ok = (ok and np.allclose(sm_params, ocl_params.get())
             and np.allclose(sm_cov_params, ocl_cov_params.get()))
    if not ok:
      print("sm.OLS")
      print(sm_params, sm_cov_params)
      print("opencl")
      print(ocl_params, ocl_cov_params)
      break
  return ok

runs = 100
print("Validating {} random runs block size 1...".format(runs))
print(validate(runs, 1))
print("Validating {} random runs block size 2...".format(runs))
print(validate(runs, 2))
print("Validating {} random runs block size 3...".format(runs))
print(validate(runs, 3))
