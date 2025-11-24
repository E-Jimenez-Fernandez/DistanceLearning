#' Initial unweighted p-norm composite index
#'
#' Computes the initial composite indicator for each statistical unit
#' using the unweighted \eqn{p}-norm of its indicator vector.
#' This function provides the starting point for the DL2 fixed-point
#' algorithm, where subsequent iterations incorporate data-driven weights.
#'
#' @param Z A numeric matrix or data frame where rows represent units
#'   (e.g., countries, regions, individuals) and columns represent individual
#'   indicators.
#' @param p A positive numeric value specifying the order of the norm.
#'   The default is \eqn{p = 2}, corresponding to the Euclidean norm.
#'
#' @return A numeric vector of length equal to the number of rows of \code{Z},
#'   where each element contains the unweighted \eqn{p}-norm value for that unit.
#'
#' @details
#' The initial index is defined as:
#' \deqn{
#'   CI^{(0)}_i = \left( \sum_{j=1}^m |Z_{ij}|^p \right)^{1/p},
#' }
#' which serves as the baseline estimate before applying machine learning
#' weighting schemes.
#'
#' @examples
#' Z <- matrix(runif(20), nrow = 5)
#' initial_index_pnorm(Z, p = 2)
#'
#' @export
initial_index_pnorm <- function(Z, p = 2) {
  Z <- as.matrix(Z)
  apply(Z, 1, function(x) (sum(abs(x)^p))^(1/p))
}
