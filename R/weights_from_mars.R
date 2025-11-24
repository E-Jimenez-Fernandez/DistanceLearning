#' Machine-learning weights using MARS (Multivariate Adaptive Regression Splines)
#'
#' Computes a vector of indicator weights using variable importance measures
#' derived from a MARS regression model, where the initial composite indicator
#' \code{ci_init} is regressed on the individual indicators. This function
#' provides an alternative, highly flexible, non-parametric weighting mechanism
#' within the DL2 framework.
#'
#' @param Z A numeric matrix or data frame containing the indicators. Rows
#'   represent statistical units and columns correspond to individual indicators.
#' @param ci_init A numeric vector containing the initial composite indicator,
#'   typically obtained from an unweighted \eqn{p}-norm or a previous iteration
#'   of the DL2 fixed-point algorithm.
#' @param epsilon A small positive constant used to guarantee strictly positive
#'   weights and prevent numerical instability. Defaults to \code{1e-6}.
#' @param ... Additional arguments passed to \code{earth::earth()}.
#'
#' @return A numeric vector of normalized weights (strictly positive and summing
#'   to one), with one weight per column of \code{Z}.
#'
#' @details
#' The function fits a MARS model via \code{earth::earth()} using the formula:
#' \deqn{
#'   \text{Compind} = f(X_1, X_2, \ldots, X_m),
#' }
#' where MARS adaptively selects hinge functions and interactions to capture
#' nonlinear relationships among indicators.
#'
#' Variable importance is extracted using \code{earth::evimp()}, and the
#' "GCV" metric (generalized cross-validation decrease) is used as the
#' importance measure:
#' \deqn{
#'   w_j \propto \max(\text{GCV}_j, \epsilon).
#' }
#'
#' Weights are then normalized to sum to one:
#' \deqn{
#'   w_j = \frac{\max(\text{GCV}_j, \epsilon)}{\sum_{k=1}^m \max(\text{GCV}_k, \epsilon)}.
#' }
#'
#' MARS provides a flexible alternative to Random Forests because it captures
#' piecewise-linear structures, nonlinearities, and interactions while
#' retaining interpretability and robustness in continuous indicator spaces.
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' Z <- matrix(runif(50), nrow = 10)
#' ci0 <- initial_index_pnorm(Z)
#' w_mars <- weights_from_mars(Z, ci_init = ci0)
#' w_mars
#' }
#'
#' @export
weights_from_mars <- function(Z, ci_init, epsilon = 1e-6, ...) {
  if (!requireNamespace("earth", quietly = TRUE)) {
    stop("Please install the 'earth' package.")
  }

  Z <- as.data.frame(Z)
  df <- data.frame(Compind = ci_init, Z)

  mars_fit <- earth::earth(
    Compind ~ .,
    data    = df,
    pmethod = "backward",
    nprune  = NULL,
    ...
  )

  imp <- earth::evimp(mars_fit)

  vars <- colnames(Z)
  importance <- numeric(length(vars))
  names(importance) <- vars

  for (v in vars) {
    if (v %in% rownames(imp)) {
      row <- imp[v, , drop = FALSE]
      importance[v] <- row[1, "gcv"]
    } else {
      importance[v] <- epsilon
    }
  }

  importance <- pmax(importance, epsilon)
  w <- importance / sum(importance)
  w
}
