#' Machine-learning weights using neural networks
#'
#' Computes a vector of indicator weights using a feed-forward neural network
#' trained to predict the initial composite indicator \code{ci_init} from the
#' individual indicators. This approach provides a nonlinear, data-driven
#' weighting scheme within the DL2 methodology, complementing Random Forest
#' and MARS alternatives.
#'
#' @param Z A numeric matrix or data frame where each row represents a unit
#'   and each column corresponds to an indicator.
#' @param ci_init A numeric vector representing the initial composite indicator
#'   for each unit. Typically produced by \code{initial_index_pnorm()} or a
#'   previous iteration of the DL2 fixed-point algorithm.
#' @param epsilon A small positive constant used to guarantee strictly positive
#'   weights and prevent numerical instability. Defaults to \code{1e-6}.
#' @param size Integer specifying the number of hidden units in the neural
#'   network. Defaults to \code{3}.
#' @param maxit Maximum number of iterations (epochs) for training the network.
#'   Defaults to \code{200}.
#' @param ... Additional arguments passed to \code{nnet::nnet()}.
#'
#' @return A numeric vector of strictly positive, normalized weights of length
#'   equal to the number of indicators. The weights sum to one.
#'
#' @details
#' The function fits a neural network using \code{nnet::nnet()} with a single
#' hidden layer and linear output activation (\code{linout = TRUE}). The raw
#' importance of each indicator is derived from the absolute values of the
#' learned network weights:
#'
#' \deqn{
#'   w_j \propto \max(|\theta_j|, \epsilon),
#' }
#'
#' where \eqn{\theta_j} denotes the connection weight associated with indicator
#' \eqn{j}.
#'
#' Since the internal representation of weights in \code{nnet::nnet()} is a
#' flattened vector, the function extracts the elements corresponding to the
#' input layer. If the number of extracted elements does not match the number
#' of indicators, padding or truncation is applied to ensure dimensional
#' consistency.
#'
#' Neural network weights capture nonlinear feature contributions and potential
#' interactions among indicators, offering a flexible alternative to tree-based
#' and spline-based methods in the DL2 weighting scheme.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' Z <- matrix(runif(40), nrow = 8)
#' ci0 <- initial_index_pnorm(Z)
#' w_nn <- weights_from_nn(Z, ci_init = ci0, size = 4)
#' w_nn
#' }
#'
#' @export
weights_from_nn <- function(Z, ci_init, epsilon = 1e-6, size = 3, maxit = 200, ...) {
  if (!requireNamespace("nnet", quietly = TRUE)) {
    stop("Please install the 'nnet' package.")
  }

  Z <- as.data.frame(Z)
  df <- data.frame(Compind = ci_init, Z)

  nn_fit <- nnet::nnet(
    Compind ~ .,
    data   = df,
    size   = size,
    linout = TRUE,
    maxit  = maxit,
    trace  = FALSE,
    ...
  )

  raw_w <- abs(nn_fit$wts)
  raw_w <- raw_w[-1]  # remove bias term

  if (length(raw_w) > ncol(Z)) {
    raw_w <- raw_w[1:ncol(Z)]
  } else if (length(raw_w) < ncol(Z)) {
    raw_w <- c(raw_w, rep(epsilon, ncol(Z) - length(raw_w)))
  }

  raw_w <- pmax(raw_w, epsilon)
  w <- raw_w / sum(raw_w)
  w
}
