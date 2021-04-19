import "../lib/github.com/nhey/ols/ols"

module ols = mk_ols f64

-- ==
-- entry: fit
-- input { 2i64 [[1.0f64, 86.0f64], [1.0f64, 76.0f64], [1.0f64, 92.0f64],
--               [1.0f64, 90.0f64], [1.0f64, 86.0f64], [1.0f64, 84.0f64],
--               [1.0f64, 93.0f64], [1.0f64, 100.0f64], [1.0f64, 87.0f64],
--               [1.0f64, 86.0f64], [1.0f64, 74.0f64], [1.0f64, 98.0f64],
--               [1.0f64, 97.0f64], [1.0f64, 84.0f64], [1.0f64, 91.0f64],
--               [1.0f64, 34.0f64], [1.0f64, 45.0f64], [1.0f64, 56.0f64],
--               [1.0f64, 44.0f64], [1.0f64, 82.0f64], [1.0f64, 72.0f64],
--               [1.0f64, 55.0f64], [1.0f64, 71.0f64], [1.0f64, 50.0f64],
--               [1.0f64, 23.0f64], [1.0f64, 39.0f64], [1.0f64, 28.0f64],
--               [1.0f64, 32.0f64], [1.0f64, 22.0f64], [1.0f64, 25.0f64],
--               [1.0f64, 29.0f64], [1.0f64, 7.0f64],  [1.0f64, 26.0f64],
--               [1.0f64, 19.0f64], [1.0f64, 15.0f64], [1.0f64, 20.0f64],
--               [1.0f64, 26.0f64], [1.0f64, 28.0f64], [1.0f64, 17.0f64],
--               [1.0f64, 22.0f64], [1.0f64, 30.0f64], [1.0f64, 25.0f64],
--               [1.0f64, 20.0f64], [1.0f64, 47.0f64], [1.0f64, 32.0f64]]
--              [62.0f64, 72.0f64, 75.0f64, 55.0f64, 64.0f64, 21.0f64,
--               64.0f64, 80.0f64, 67.0f64, 72.0f64, 42.0f64, 76.0f64,
--               76.0f64, 41.0f64, 48.0f64, 76.0f64, 53.0f64, 60.0f64,
--               42.0f64, 78.0f64, 29.0f64, 48.0f64, 55.0f64, 29.0f64,
--               21.0f64, 47.0f64, 81.0f64, 36.0f64, 22.0f64, 44.0f64,
--               15.0f64,  7.0f64, 42.0f64,  9.0f64, 21.0f64, 21.0f64,
--               16.0f64, 16.0f64,  9.0f64, 14.0f64, 12.0f64, 17.0f64,
--                7.0f64, 34.0f64,  8.0f64] }
-- output { [10.60349832f64, 0.59485944f64]
--          [[9.30974511e-02f64, -1.34857729e-03f64],
--           [-1.34857729e-03f64, 2.56600331e-05f64]]
--          2i64 }
entry fit [m][n] (bsz: i64) (X: [m][n]f64) (y: [m]f64) =
  let res = ols.fit bsz X y
  in (res.params, res.cov_params, res.rank)

-- Test linear dependency detection; rank is less than the number of parameters.
-- ==
-- entry: fit_rank
-- input { 1i64
--         [[1.0f64, 1.0f64,   0.5f64,       0.866025f64,  0.866025f64,  0.5f64,
--            1.0f64,  0.0f64],
--          [1.0f64, 2.0f64,   0.866025f64,  0.5f64,       0.866025f64, -0.5f64,
--            0.0f64, -1.0f64],
--          [1.0f64, 4.0f64,   0.866025f64, -0.5f64,      -0.866025f64, -0.5f64,
--           -0.0f64,  1.0f64],
--          [1.0f64, 5.0f64,   0.5f64,      -0.866025f64, -0.866025f64,  0.5f64,
--            1.0f64,  0.0f64],
--          [1.0f64, 7.0f64,  -0.5f64,      -0.866025f64,  0.866025f64,  0.5f64,
--           -1.0f64, -0.0f64],
--          [1.0f64, 9.0f64,  -1.0f64,      -0.0f64,       0.0f64,      -1.0f64,
--            1.0f64,  0.0f64],
--          [1.0f64, 14.0f64,  0.866025f64,  0.5f64,       0.866025f64, -0.5f64,
--            0.0f64, -1.0f64],
--          [1.0f64, 17.0f64,  0.5f64,      -0.866025f64, -0.866025f64,  0.5f64,
--            1.0f64,  0.0f64]]
--         [4074.389219f64, 6842.004064f64, 6378.106146f64, 5561.734381f64,
--          6480.266660f64, 4951.876764f64, 4043.239055f64, 7192.433631f64] }
-- output { 7i64 }
entry fit_rank b X y = (fit b X y).2
