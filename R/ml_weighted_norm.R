#' Machine-learning weighted p-norm composite indicator
#'
#' Computes a weighted \eqn{p}-norm for each statistical unit, where the
#' weights are typically obtained from machine-learning procedures such as
#' random forests, MARS, or neural networks. This function forms the core of the
#' DL2 composite indicator, translating a matrix of indicator values and a
#' weight vector into a synthetic distance-based index.
#'
#' @param Z A numeric matrix or data frame of indicator values. Each row
#'   corresponds to a statistical unit and each column corresponds to an
#'   individual indicator.
#' @param weights A numeric vector of non-negative weights. Its length must
#'   match the number of columns of \code{Z}. The weights are usually obtained
#'   from one of the DL2 weight-estimation functions such as
#'   \code{\link{weights_from_rf}}, \code{\link{weights_from_mars}}, or
#'   \code{\link{weights_from_nn}}.
#' @param p A positive real number indicating the order of the norm. The default
#'   is \code{2}, corresponding to a weighted Euclidean distance. Values of
#'   \code{p < 1} are not allowed.
#'
#' @return A numeric vector of length equal to the number of rows of \code{Z},
#'   containing the DL2 weighted \eqn{p}-norm composite indicator for each unit.
#'
#' @details
#' The function computes, for each statistical unit \eqn{i}, the expression
#' \deqn{
#'   CI_i = \left( \sum_{j=1}^m w_j |Z_{ij}|^p \right)^{1/p},
#' }
#' where \eqn{w_j} are the weights and \eqn{Z_{ij}} are the indicator values.
#' The function performs basic dimensionality checks and converts inputs to the
#' appropriate numeric form.
#'
#' This function is used iteratively within
#' \code{\link{ci_ml_fixedpoint}} to compute the DL2 composite indicator during
#' the fixed-point procedure. It is also used in robustness analyses such as
#' \code{\link{robust_ci_vs_weights}}.
#'
#' @examples
#' Z <- matrix(runif(20), nrow = 5)
#' w <- c(0.2, 0.3, 0.5,0.4)
#' ml_weighted_norm(Z, weights = w, p = 2)
#'
#' @export
ml_weighted_norm <- function(Z, weights, p = 2) {
  Z <- as.matrix(Z)
  weights <- as.numeric(weights)

  if (length(weights) != ncol(Z)) {
    stop("Length of 'weights' must match number of columns of Z.")
  }

  apply(Z, 1, function(x) (sum(weights * abs(x)^p))^(1/p))
}
