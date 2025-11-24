#' Machine-learning weights using Random Forests
#'
#' Computes a vector of indicator weights using the variable importance
#' derived from a Random Forest regression model, where the initial
#' composite indicator \code{ci_init} is regressed on the individual
#' indicators. This weighting scheme forms one of the core components
#' of the DL2 methodology for constructing composite indicators with
#' machine-learning-based aggregation rules.
#'
#' @param Z A numeric matrix or data frame of indicators where rows represent
#'   units and columns represent individual indicators.
#' @param ci_init A numeric vector representing the initial composite indicator
#'   (typically obtained using an unweighted \eqn{p}-norm or a previous iteration
#'   of the DL2 algorithm).
#' @param ntree Integer indicating the number of trees in the Random Forest.
#'   Defaults to \code{500}.
#' @param epsilon A small positive constant guaranteeing strictly positive
#'   weights and preventing numerical instability. Defaults to \code{1e-6}.
#' @param ... Additional arguments passed to \code{randomForest::randomForest()}.
#'
#' @return A numeric vector of normalized weights of length equal to the number
#'   of indicators (columns of \code{Z}). The weights sum to one and are strictly
#'   positive due to the use of \code{epsilon}.
#'
#' @details
#' The weighting scheme is based on the Random Forest variable importance measure
#' (IncMSE), extracted via
#' \code{randomForest::importance(type = 1)}. For each indicator \eqn{X_j}, its
#' importance reflects the reduction in prediction error when the variable is
#' used in the ensemble model.
#'
#' The raw importances may contain zeros or missing entries (e.g., when predictors
#' are collinear or unused); thus, the function applies:
#' \deqn{
#' w_j = \max(\text{importance}_j, \epsilon),
#' }
#' followed by normalization to ensure:
#' \deqn{
#' \sum_j w_j = 1.
#' }
#'
#' This step provides a non-parametric, data-driven weighting mechanism
#' that captures nonlinearities and interactions in the indicators, and
#' therefore complements the DL2 norm-based aggregation framework.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' Z <- matrix(runif(50), nrow = 10)
#' ci0 <- initial_index_pnorm(Z, p = 2)
#' w_rf <- weights_from_rf(Z, ci_init = ci0, ntree = 200)
#' w_rf
#' }
#'
#' @export
weights_from_rf <- function(Z, ci_init, ntree = 500, epsilon = 1e-6, ...) {
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("Please install the 'randomForest' package.")
  }

  Z <- as.data.frame(Z)
  df <- data.frame(Compind = ci_init, Z)

  rf_fit <- randomForest::randomForest(
    Compind ~ .,
    data   = df,
    ntree  = ntree,
    importance = TRUE,
    ...
  )

  imp <- randomForest::importance(rf_fit, type = 1)

  vars <- colnames(Z)
  importance <- numeric(length(vars))
  names(importance) <- vars

  for (v in vars) {
    if (v %in% rownames(imp)) {
      importance[v] <- imp[v, 1]
    } else {
      importance[v] <- epsilon
    }
  }

  importance <- pmax(importance, epsilon)
  w <- importance / sum(importance)
  w
}
