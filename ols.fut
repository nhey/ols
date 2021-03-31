-- | Implementation of ordinary least squares (OLS) regression
-- using QR decomposition.
-- 
-- This module makes use of work by Kasper Unn Weihe,
-- Kristian Quirin Hansen and Peter Kanstrup Larsen. Several
-- internal functions have been copied from [their
-- report](https://futhark-lang.org/student-projects/kristian-kasper-peter-project.pdf).

import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/diku-dk/linalg/qr"


module type ols = {
  type t
  -- | `params` are the estimated coefficients (β) and
  -- `cov_params` is the covariance matrix `(X^T X)^{-1}`
  type ols_result [n] = { params: [n]t, cov_params: [n][n]t }
  -- | Ordinary Least Squares (OLS) for estimating parameters
  -- in a linear regression model.
  -- The linear least squares equations for regression, `X^T X b = X^T y`,
  -- is solved using QR decomposition.
  --
  -- See [mk_block_householder](https://futhark-lang.org/pkgs/github.com/diku-dk/linalg/0.4.1/) for important details on `block_size`.
  val ols [m][n] : (block_size: i64) -> (X: [m][n]t) -> (y: [m]t) -> ols_result [n]
}

-- TODO: why double transpose in cho_inv, cho_inv2?
module mk_ols (T: real): ols with t = T.t = {
  module linalg = mk_linalg T
  module block_householder = mk_block_householder T

  type t = T.t

  -- This is also defined in the linalg module, but we need
  -- need to make it clear that the result does not alias its
  -- input. Otherwise copying must take place during in-place
  -- updates in `forward_substitution` and `back_substitution`.
  -- With `T` being reals, `T.t` is a scalar so this is safe.
  let dotprod [n] (xs: [n]t) (ys: [n]t): *t =
    T.(reduce (+) (i64 0) (map2 (*) xs ys))

  let identity (n: i64): [n][n]t =
    tabulate_2d n n (\i j ->if j == i then T.i64 1 else T.i64 0)

  let forward_substitution [n] (L: [n][n]t) (b: [n]t): [n]t =
    let y = replicate n (T.i64 0)
    in loop y for i in 0..<n do
      let sumy = dotprod L[i,:i] y[:i]
      let y[i] = (b[i] T.- sumy) T./ L[i,i]
      in y

  let back_substitution [n] (U: [n][n]t) (y: [n]t): [n]t =
    let x = replicate n (T.i64 0)
    in loop x for j in 0..<n do
      let i = n - j - 1
      let sumx = dotprod U[i,i+1:n] x[i+1:n]
      let x[i] = (y[i] T.- sumx) T./ U[i,i]
      in x

  -- Compute `(U^T U)^{-1} = U^{-1} (U^T)^{-1} = U^{-1} (U^{-1})^T`.
  -- Should be the same as `cho_inv`, but using the upper triangular
  -- matrix. If fed `R` from QR decomposition of `X`, result is
  -- `(X^T X)^{-1}`; thanks to hint given by R function of same name.
  let chol2inv [n] (U: [n][n]t): [n][n]t =
    let UinvT = map (back_substitution U) (identity n)
    in linalg.matmul (transpose UinvT) UinvT

  type ols_result [n] = { params: [n]t, cov_params: [n][n]t }

  let ols [m][n] (bsz: i64) (X: [m][n]t) (y: [m]t): ols_result [n] =
    let (Q, R) = block_householder.qr bsz X
    -- The shared dimension is `k = min(m,n)`. However fitting a
    -- linear regression with `n` parameters requires at least
    -- `n` datapoints, so we always have `m >= n`; consequently `k = n`.
    let Q = Q[:m,:n]
    let R = R[:n,:n]
    let cov_params = chol2inv R
    -- Solve `Xb = y`. Given `X = QR` and `Q^T Q = I`, we have
    -- `Xb = y => (QR)b = y => Rb = Q^T y`. This last equation
    -- can be solved using back substitution since R is upper
    -- triangular.
    let effects = linalg.matvecmul_row (transpose Q) y
    let beta = back_substitution R effects
    in { params = beta, cov_params = cov_params }
}
