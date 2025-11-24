#' Comparison of DL2 composite indicators across machine-learning weighting methods
#'
#' Computes and compares the final DL2 composite indicators obtained using
#' three alternative machine-learning methods for weight estimation:
#' Random Forests, MARS, and neural networks. The function evaluates the
#' degree of concordance between the resulting indices using Spearman,
#' Kendall, and Pearson correlations. This provides an empirical assessment
#' of the sensitivity of DL2 results to the choice of weighting methodology.
#'
#' @param Z A numeric matrix or data frame of indicators where rows represent
#'   units and columns correspond to individual indicators.
#' @param p Numeric value specifying the order of the norm used in the DL2
#'   composite indicator. Defaults to \code{2}.
#' @param ntree Integer indicating the number of trees used when
#'   \code{weight_method = "rf"} (Random Forests). Defaults to \code{500}.
#' @param max_iter Maximum number of iterations in the DL2 fixed-point algorithm.
#'   Defaults to \code{50}.
#' @param lambda Damping parameter used to update the weights iteratively.
#'   Must lie in \eqn{(0,1)}. Defaults to \code{0.2}.
#' @param n_stable Integer specifying how many consecutive iterations must fall
#'   below the internal tolerance threshold to declare convergence. Defaults to
#'   \code{3}.
#' @param seed Optional integer seed for reproducibility. Defaults to \code{NULL}.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{RF_vs_MARS}: List of Spearman, Kendall, and Pearson correlations
#'     between the DL2 indices computed with Random Forests and MARS weights.
#'   \item \code{RF_vs_NN}: Same comparisons between Random Forests and neural networks.
#'   \item \code{MARS_vs_NN}: Comparisons between MARS and neural network weighting.
#'   \item \code{ci_rf}: Final DL2 index using Random Forest weights.
#'   \item \code{ci_mars}: Final DL2 index using MARS weights.
#'   \item \code{ci_nn}: Final DL2 index using neural network weights.
#' }
#'
#' @details
#' The function runs the \code{ci_ml_fixedpoint()} algorithm three times,
#' once using each machine-learning method to estimate the weights. It then
#' evaluates the agreement between the resulting composite indicators using
#' three classical rank or linear concordance measures:
#'
#' \itemize{
#'   \item \strong{Spearman}: monotonic ranking similarity,
#'   \item \strong{Kendall}: pairwise ordinal consistency,
#'   \item \strong{Pearson}: linear correlation.
#' }
#'
#' These measures help assess the robustness of DL2 with respect to the choice
#' of weighting algorithm, guiding informed decisions when selecting the most
#' appropriate machine-learning method.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' Z <- matrix(runif(80), nrow = 16)
#'
#' cmp <- compare_ml_methods_ci(Z, p = 2)
#' cmp$RF_vs_MARS
#' cmp$RF_vs_NN
#' cmp$MARS_vs_NN
#' }
#'
#' @export
compare_ml_methods_ci <- function(
    Z,
    p        = 2,
    ntree    = 500,
    max_iter = 50,
    lambda   = 0.2,
    n_stable = 3,
    seed     = NULL
) {

  if (!is.null(seed)) set.seed(seed)

  # ---- Run DL2 with each weighting method ----
  res_rf   <- ci_ml_fixedpoint(Z, p = p, ntree = ntree, max_iter = max_iter,
                               lambda = lambda, n_stable = n_stable,
                               seed = seed, verbose = FALSE,
                               weight_method = "rf")

  res_mars <- ci_ml_fixedpoint(Z, p = p, ntree = ntree, max_iter = max_iter,
                               lambda = lambda, n_stable = n_stable,
                               seed = seed, verbose = FALSE,
                               weight_method = "mars")

  res_nn   <- ci_ml_fixedpoint(Z, p = p, ntree = ntree, max_iter = max_iter,
                               lambda = lambda, n_stable = n_stable,
                               seed = seed, verbose = FALSE,
                               weight_method = "nn")

  ci_rf   <- res_rf$ci_final
  ci_mars <- res_mars$ci_final
  ci_nn   <- res_nn$ci_final

  # --- auxiliary function for pairwise comparison ---
  compare_two <- function(a, b) {
    list(
      spearman = cor(a, b, method = "spearman"),
      kendall  = cor(a, b, method = "kendall"),
      pearson  = cor(a, b, method = "pearson")
    )
  }

  list(
    RF_vs_MARS = compare_two(ci_rf, ci_mars),
    RF_vs_NN   = compare_two(ci_rf, ci_nn),
    MARS_vs_NN = compare_two(ci_mars, ci_nn),
    ci_rf      = ci_rf,
    ci_mars    = ci_mars,
    ci_nn      = ci_nn
  )
}
