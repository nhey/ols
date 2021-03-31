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

-- TODO: extend result
-- TODO: why double transpose in cho_inv, cho_inv2?
-- TODO: get actual rank of R and resize to square matrix, cho_inv, cho_inv2
module mk_ols (T: real): ols with t = T.t = {
  module linalg = mk_linalg T
  module block_householder = mk_block_householder T

  type t = T.t

  -- This is also defined in the linalg module, but we need
  -- need to make it clear that the result does not alias its
  -- input. Otherwise copying must take place during in-place
  -- updates in `forward_substitution` and `back_substitution`.
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

  -- TODO: try replacing with back subst and not transposing R?
  let cho_inv2 [n] (L: [n][n]t): [n][n]t =
    let Linv = map (forward_substitution L) (identity n) |> transpose
    in linalg.matmul (transpose Linv) Linv

  type ols_result [n] = { params: [n]t, cov_params: [n][n]t }

  let ols [m][n] (bsz: i64) (X: [m][n]t) (y: [m]t): ols_result [n] =
    let (Q, R) = block_householder.qr bsz X
    let Q = Q[:m,:n] -- TODO: n should be min(m,n)
    let R = R[:n,:n] -- TODO: first n should be min(m,n)
    let cov_params = cho_inv2 (transpose R)
    -- Solve Xb = y => (QR)b = y => Rb = (Q.T) y since Q.T Q = I.
    -- The last equation can be solved using back substitution
    -- since R is upper triangular.
    let effects = linalg.matvecmul_row (transpose Q) y
    let beta = back_substitution R effects
    in { params = beta, cov_params = cov_params }
}
