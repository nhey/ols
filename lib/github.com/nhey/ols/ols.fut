-- | Implementation of ordinary least squares (OLS) regression
-- using QR decomposition.
-- 
-- This module makes use of work by Kasper Unn Weihe,
-- Kristian Quirin Hansen and Peter Kanstrup Larsen. Several
-- internal functions have been copied from [their
-- report](https://futhark-lang.org/student-projects/kristian-kasper-peter-project.pdf).

module type ols = {
  type t
  -- | `params` are the estimated coefficients (β),
  -- `cov_params` is the covariance matrix `(X^T X)^{-1}` and
  -- `rank` denotes the rank of the regressor matrix `X`.
  -- If `rank` is less than the number of parameters in `X`,
  -- the OLS estimator does not exist and the output should
  -- not be treated as OLS results. If you run into this,
  -- a suggestion is to drop the offending parameters from `X`.
  -- See [multicolinearity](https://en.wikipedia.org/wiki/Multicollinearity).
  type results [n] = { params: [n]t, cov_params: [n][n]t, rank: i64 }
  -- | Ordinary Least Squares (OLS) for estimating parameters
  -- in a linear regression model.
  -- The linear least squares equations for regression, `X^T X b = X^T y`,
  -- is solved using QR decomposition.
  --
  -- See [mk_block_householder](https://futhark-lang.org/pkgs/github.com/diku-dk/linalg/0.4.1/)
  -- for important details on `block_size`.
  val fit [m][n] : (block_size: i64) -> (X: [m][n]t) -> (y: [m]t) -> results [n]
}

module mk_ols (T: float): ols with t = T.t = {
  import "../../diku-dk/linalg/linalg"
  import "../../diku-dk/linalg/qr"

  module linalg = mk_linalg T
  module block_householder = mk_block_householder T

  type t = T.t

  -- This is also defined in the linalg module, but we need
  -- need to make it clear that the result does not alias its
  -- input. Otherwise copying must take place during in-place
  -- updates in `forward_substitution` and `back_substitution`.
  -- `T.t` is a scalar when `T` is the reals, so this is safe.
  let dotprod [n] (xs: [n]t) (ys: [n]t): *t =
    T.(reduce (+) (i64 0) (map2 (*) xs ys))

  -- Copied from report on linear algebra in Futhark.
  let identity (n: i64): [n][n]t =
    tabulate_2d n n (\i j ->if j == i then T.i64 1 else T.i64 0)

  -- Copied from report on linear algebra in Futhark.
  let forward_substitution [n] (L: [n][n]t) (b: [n]t): [n]t =
    let y = replicate n (T.i64 0)
    in loop y for i in 0..<n do
      let sumy = dotprod L[i,:i] y[:i]
      let y[i] = (b[i] T.- sumy) T./ L[i,i]
      in y

  -- Copied from report on linear algebra in Futhark.
  let back_substitution [n] (U: [n][n]t) (y: [n]t): [n]t =
    let x = replicate n (T.i64 0)
    in loop x for j in 0..<n do
      let i = n - j - 1
      let sumx = dotprod U[i,i+1:n] x[i+1:n]
      let x[i] = (y[i] T.- sumx) T./ U[i,i]
      in x

  -- Given an upper triangular matrix `U`, compute `(U^T U)^{-1}`.
  -- Also returns the intermediate result `U^{-1}`.
  -- * If fed transpose of `L` from Cholesky decomposition of a symmetric,
  -- positive definite square matrix `A`, result is `A^{-1}`.
  -- * If fed `R` from QR decomposition of `X`, result is `(X^T X)^{-1}`
  -- since `X^T X = R^T Q^T QR = R^T R`.
  -- The name is an homage to similar functionality in the R language.
  let chol2inv [n] (U: [n][n]t): ([n][n]t, [n][n]t) =
    let UinvT = map (back_substitution U) (identity n)
    let Uinv = transpose UinvT
    -- Compute `(U^T U)^{-1} = U^{-1} (U^T)^{-1} = U^{-1} (U^{-1})^T`.
    in (linalg.matmul Uinv UinvT, Uinv)

  -- Compute rank of `X` as number of non-zero elements in the
  -- leading diagonal of `R` from QR decomposition of `X`.
  -- Where non-zeros are greater than `t.abs R[0,0] * n * t.epsilon`.
  --
  -- NOTE: This is not entirely correct; it would require a
  -- "rank-revealing" QR factorisation. Which is not available,
  -- so this will have to suffice. It does however validate when
  -- using a large epsilon value.
  let matrix_rank [m][n] (R: [m][n]t): i64 =
    -- TODO tol is 1e3 too large
    let tol = T.(abs R[0,0] * i64 n * epsilon * f32 1e4)
    in map (\i -> i64.bool T.(abs R[i,i] > tol)) (iota n) |> i64.sum

  type results [n] = { params: [n]t, cov_params: [n][n]t, rank: i64 }

  let fit [m][n] (bsz: i64) (X: [m][n]t) (y: [m]t): results [n] =
    let (Q, R) = block_householder.qr bsz X
    let rank = matrix_rank R
    -- The shared dimension is `k = min(m,n)`. However fitting a
    -- linear regression with `n` parameters requires at least
    -- `n` datapoints, so we always have `m >= n`; consequently `k = n`.
    let Q = Q[:m,:n]
    let R = R[:n,:n]
    let (cov_params, Rinv) = chol2inv R
    -- Find least squares solution to `Xb = y`. Substituting
    -- `X = QR` into the LLS equations `X^T X b = X^T y`, we get
    -- `((QR)^T QR) b = (QR)^T y <=> (R^T Q^T QR) b = R^T Q^T y`.
    -- Now since `Q^T Q = I`, we have `R^T R b = R^T Q^T y`.
    -- Here, we can ignore the `R^T` factor yielding `R b = Q^T y`.
    -- With `R` being upper triangular, this last equation may be
    -- solved using back substitution. But we already have
    -- `R^{-1}` so premultiplying this is faster.
    let effects = linalg.matvecmul_row (transpose Q) y
    let beta = linalg.matvecmul_row Rinv effects
    in { params = beta, cov_params = cov_params, rank = rank }

  -- TODO: benchmark whether this is ever worth it.
  let fit_reduced [m][n] (bsz: i64) (X: [m][n]t) (y: [m]t): ([n]t, i64) =
    let (Q, R) = block_householder.qr bsz X
    let rank = matrix_rank R
    let Q = Q[:m,:n]
    let R = R[:n,:n]
    let effects = linalg.matvecmul_row (transpose Q) y
    let beta = back_substitution R effects
    in (beta, rank)
}
