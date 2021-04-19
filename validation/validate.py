import statsmodels.api as sm
import numpy as np
import scipy
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

ocl_params, ocl_cov_params, ocl_rank = ols_fut.fit(1, X, y)

print("Validating statsmodels example")
print("\n Covariance of parameters")
print("manual:", manual_cov_params)
print("sm.OLS:", results.normalized_cov_params)
print("opencl:", ocl_cov_params.get())

print(" Parameters")
print("sm.OLS:", results.params)
print("opencl:", ocl_params)

print(" Rank of regressor matrix")
print("sm.OLS:", model.rank)
print("opencl:", ocl_rank)

def validate(num_tests, bsz, rank_deficient=False):
  ok = True
  for i in range(num_tests):
    X = np.random.rand(100,6).astype(np.float64)
    y = np.random.rand(100,1).astype(np.float64)
    model = sm.OLS(y,X)
    sm_results = model.fit(method="qr")
    sm_params = sm_results.params
    sm_cov_params = sm_results.normalized_cov_params
    sm_rank = model.rank

    ocl_params, ocl_cov_params, ocl_rank = ols_fut.fit(bsz, X, y)

    ok = (ok and np.allclose(sm_params, ocl_params.get())
             and np.allclose(sm_cov_params, ocl_cov_params.get())
             and sm_rank == ocl_rank)
    if not ok:
      print("sm.OLS")
      print(sm_params, sm_cov_params)
      print("opencl")
      print(ocl_params, ocl_cov_params)
      break
  return ok

runs = 1000
print("\nValidating {} random runs block size 1...".format(runs))
print(validate(runs, 1))
print("Validating {} random runs block size 2...".format(runs))
print(validate(int(runs/2), 2))
print("Validating {} random runs block size 3...".format(runs))
print(validate(int(runs/2), 3))

def validate_rank_deficient(num_tests, bsz, rank_deficient=False):
  ok = True
  tests = 0
  while tests < num_tests:
    X = np.random.rand(100,6).astype(np.float64)
    y = np.random.rand(100,1).astype(np.float64)
    # Make X rank deficient
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    zero_inds = np.random.choice(6)
    s[zero_inds] = 0.0
    X = (u * s) @ vh
    # Singular matrices will produce errors in statsmodels.
    # "Near singular" matrices do not and will produce bogus
    # results in statsmodels, silently.
    # It is supposedly detectable by looking at the rank of X.
    # This is what the futhark library mimics.
    model = sm.OLS(y,X)
    try:
      sm_results = model.fit(method="qr")
    except:
      continue
    sm_rank = model.rank
    Q, R = np.linalg.qr(X)

    _, _, ocl_rank = ols_fut.fit(bsz, X, y)

    tests += 1
    check = sm_rank == ocl_rank
    if not check:
      print("sm.OLS", sm_rank)
      print("opencl", ocl_rank)
      print(np.diag(R))
      ok = False
  return ok

runs = 1000
print("\nValidating {} random runs block size 1...".format(runs))
print(validate_rank_deficient(runs, 1))

