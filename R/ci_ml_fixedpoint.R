#' DL2 fixed-point algorithm for composite indicators with machine-learning weights
#'
#' Implements the core DL2 fixed-point algorithm to compute a composite
#' indicator using a weighted \eqn{p}-norm, where the weights are updated
#' iteratively using machine-learning techniques. At each iteration, a new set
#' of weights is estimated from a predictive model (Random Forests, MARS, or
#' neural networks) trained to approximate the current composite index. The
#' updated weights are combined with those from the previous iteration using a
#' damping parameter, and the process continues until convergence.
#'
#' @param Z A numeric matrix or data frame of indicators, where rows represent
#'   units and columns represent individual indicators.
#' @param p A positive numeric value specifying the order of the weighted norm
#'   used to compute the composite indicator. Defaults to \code{2}.
#' @param ntree Number of trees used when \code{weight_method = "rf"} (Random Forests).
#'   Defaults to \code{500}.
#' @param max_iter Maximum number of fixed-point iterations allowed. Defaults to
#'   \code{50}.
#' @param lambda Damping (relaxation) parameter in the interval \eqn{(0,1)} used to
#'   stabilise weight updates: \deqn{ w_{t+1} = (1 - \lambda) w_t + \lambda \tilde{w}_t }.
#'   Defaults to \code{0.2}.
#' @param n_stable Number of consecutive iterations below the tolerance threshold
#'   required to declare convergence. Defaults to \code{3}.
#' @param seed Optional integer seed for reproducibility. Defaults to \code{NULL}.
#' @param verbose Logical; if \code{TRUE}, prints progress information during
#'   iterations. Defaults to \code{TRUE}.
#' @param weight_method Character string specifying the machine-learning method
#'   used to update the weights at each iteration. Must be one of:
#'   \code{"rf"} (Random Forests), \code{"mars"} (MARS splines), or \code{"nn"}
#'   (neural networks).
#' @param epsilon A small positive constant used to ensure strictly positive weights
#'   and avoid numerical instability. Defaults to \code{1e-6}.
#' @param ... Additional arguments passed to the underlying ML functions.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{ci_seq}: Matrix storing the composite indicator at each iteration.
#'   \item \code{w_seq}: Matrix storing the weight vector at each iteration.
#'   \item \code{ci_final}: Final composite indicator.
#'   \item \code{w_final}: Final weight vector.
#'   \item \code{L1_diff}: Vector of L1 differences between successive weight vectors.
#'   \item \code{tol}: Convergence tolerance used internally.
#'   \item \code{n_iter}: Number of iterations performed.
#'   \item \code{method}: Machine-learning weighting method employed.
#' }
#'
#' @details
#' The algorithm follows a classical fixed-point iteration applied to the DL2
#' composite indicator. Given an initial unweighted \eqn{p}-norm index:
#' \deqn{ CI^{(0)}_i = \left( \sum_j |Z_{ij}|^p \right)^{1/p}, }
#' a machine-learning model is trained to approximate \eqn{ CI^{(t)} }, and the
#' derived variable importance measures yield a new set of raw weights
#' \eqn{\tilde{w}_t}.
#'
#' These are regularised using:
#' \deqn{
#'   w_{t+1} = (1 - \lambda) w_t + \lambda \tilde{w}_t,
#' }
#' where \eqn{\lambda \in (0,1)} controls the smoothness of updates.
#'
#' Convergence is detected when the L1-norm of weight differences satisfies:
#' \deqn{
#'   \| w_{t+1} - w_t \|_1 < \mathrm{tol},
#' }
#' for at least \code{n_stable} consecutive iterations, with the tolerance set
#' internally to \eqn{ 0.1 / m } where \eqn{m} is the number of indicators.
#'
#' This iterative process ensures a stable representation of the DL2 metric,
#' incorporating nonlinear relationships captured by the selected ML method.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' Z <- matrix(runif(100), nrow = 20)
#'
#' # Using Random Forest weights
#' res_rf <- ci_ml_fixedpoint(Z, weight_method = "rf", verbose = FALSE)
#'
#' # Using MARS spline weights
#' res_mars <- ci_ml_fixedpoint(Z, weight_method = "mars", verbose = FALSE)
#'
#' # Using neural network weights
#' res_nn <- ci_ml_fixedpoint(Z, weight_method = "nn", size = 4, verbose = FALSE)
#'
#' res_rf$ci_final
#' res_rf$w_final
#' }
#'
#' @export
ci_ml_fixedpoint <- function(
    Z,
    p            = 2,
    ntree        = 500,
    max_iter     = 50,
    lambda       = 0.2,
    n_stable     = 3,
    seed         = NULL,
    verbose      = TRUE,
    weight_method = c("rf", "mars", "nn"),
    epsilon      = 1e-6,
    ...
) {
  weight_method <- match.arg(weight_method)
  Z <- as.matrix(Z)
  n <- nrow(Z)
  m <- ncol(Z)
  tol <- 0.1 / m
  if (!is.null(seed)) {
    set.seed(seed)
  }

  get_weights <- function(Z, ci, method, ntree, epsilon, ...) {
    if (method == "rf") {
      w <- weights_from_rf(Z, ci_init = ci, ntree = ntree, epsilon = epsilon, ...)
    } else if (method == "mars") {
      w <- weights_from_mars(Z, ci_init = ci, epsilon = epsilon, ...)
    } else if (method == "nn") {
      w <- weights_from_nn(Z, ci_init = ci, epsilon = epsilon, ...)
    } else {
      stop("Unsupported weight_method: ", method)
    }
    w
  }

  ci0 <- initial_index_pnorm(Z, p = p)
  w0  <- get_weights(Z, ci = ci0, method = weight_method,
                     ntree = ntree, epsilon = epsilon, ...)

  ci_mat  <- matrix(NA_real_, nrow = n, ncol = max_iter + 1)
  w_mat   <- matrix(NA_real_, nrow = m, ncol = max_iter + 1)
  L1_diff <- numeric(max_iter)

  ci_mat[, 1] <- ci0
  w_mat[,  1] <- w0
  w_prev  <- w0
  n_below <- 0

  for (t in seq_len(max_iter)) {
    if (verbose) {
      cat("Iteration", t, "\n")
    }

    ci_t <- ml_weighted_norm(Z, weights = w_prev, p = p)

    if (!is.null(seed)) {
      set.seed(seed + t)
    }

    w_raw <- get_weights(Z, ci = ci_t, method = weight_method,
                         ntree = ntree, epsilon = epsilon, ...)

    w_new <- (1 - lambda) * w_prev + lambda * w_raw

    diff_t <- sum(abs(w_new - w_prev))
    L1_diff[t] <- diff_t

    if (verbose) {
      cat("  L1 difference =", round(diff_t, 4), "\n")
    }

    ci_mat[, t + 1] <- ci_t
    w_mat[,  t + 1] <- w_new

    if (diff_t < tol) {
      n_below <- n_below + 1
    } else {
      n_below <- 0
    }

    if (n_below >= n_stable) {
      if (verbose) {
        cat("Convergence reached after", t, "iterations\n")
      }
      n_iter <- t
      ci_mat  <- ci_mat[, 1:(t + 1), drop = FALSE]
      w_mat   <- w_mat[,  1:(t + 1), drop = FALSE]
      L1_diff <- L1_diff[1:t]
      return(list(
        ci_seq   = ci_mat,
        w_seq    = w_mat,
        w_final  = w_new,
        ci_final = ci_t,
        L1_diff  = L1_diff,
        tol      = tol,
        n_iter   = n_iter,
        method   = weight_method
      ))
    }

    w_prev <- w_new
  }

  if (verbose) {
    cat("Maximum iterations reached without formal convergence.\n")
  }

  list(
    ci_seq   = ci_mat,
    w_seq    = w_mat,
    w_final  = w_prev,
    ci_final = ci_mat[, max_iter + 1],
    L1_diff  = L1_diff,
    tol      = tol,
    n_iter   = max_iter,
    method   = weight_method
  )
}
