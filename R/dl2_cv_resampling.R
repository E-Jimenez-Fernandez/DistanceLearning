#' Resampling-based cross-validation for DL2 composite indicators
#'
#' Performs a resampling-based validation procedure for the DL2 composite
#' indicator using bootstrap or subsampling. This function evaluates the
#' robustness of the final DL2 index and its associated weights under repeated
#' perturbations of the dataset. The method re-estimates the DL2 composite
#' indicator on multiple bootstrap samples and compares each re-estimated index
#' with the baseline index obtained from the full dataset.
#'
#' @param Z A numeric matrix or data frame of indicators, where rows represent
#'   statistical units and columns correspond to individual indicators.
#' @param weight_method Character string indicating the machine-learning method
#'   used to estimate weights. Must be one of \code{"rf"}, \code{"mars"}, or
#'   \code{"nn"}. Defaults to \code{"rf"}.
#' @param B Integer specifying the number of bootstrap or resampling replications.
#'   Defaults to \code{100}.
#' @param prop Proportion of units included in each resampling replication.
#'   Defaults to \code{0.8}. The procedure uses bootstrap sampling by default
#'   (sampling with replacement).
#' @param p Order of the norm used to compute the DL2 index. Defaults to \code{2}.
#' @param ntree Number of trees used when \code{weight_method = "rf"}.
#'   Defaults to \code{500}.
#' @param max_iter Maximum number of iterations allowed in the DL2 fixed-point
#'   algorithm. Defaults to \code{50}.
#' @param lambda Damping parameter used for weight updating in the iterative
#'   DL2 procedure. Must lie between zero and one. Defaults to \code{0.2}.
#' @param n_stable Number of consecutive iterations falling below the tolerance
#'   threshold required to declare convergence. Defaults to \code{3}.
#' @param seed Optional integer seed for reproducibility. Defaults to \code{NULL}.
#' @param verbose Logical indicating whether progress information should be
#'   displayed. Defaults to \code{TRUE}.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{method}: Weighting method used.
#'   \item \code{rho_spearman}: Vector of Spearman rank correlations between
#'     the baseline index and resampled indices.
#'   \item \code{rho_kendall}: Vector of Kendall rank correlations.
#'   \item \code{L1_w}: L1 distances between baseline and re-estimated weights.
#'   \item \code{summary_spearman}: Summary statistics of the Spearman correlations.
#'   \item \code{summary_kendall}: Summary statistics of the Kendall correlations.
#'   \item \code{summary_L1_w}: Summary statistics of the L1 weight differences.
#'   \item \code{res_base}: The full-sample DL2 estimation used as the benchmark.
#' }
#'
#' @details
#' This function implements a modular resampling strategy designed to evaluate the
#' stability and robustness of the DL2 composite indicator under repeated random
#' perturbations of the dataset. For each replication, a bootstrap sample is drawn,
#' the DL2 index is re-estimated using the chosen machine-learning method, and the
#' resulting index is compared with the baseline index computed from the full sample.
#'
#' The method quantifies robustness through:
#' \itemize{
#'   \item \strong{Spearman correlation}: Measures monotonic ranking similarity.
#'   \item \strong{Kendall correlation}: Measures pairwise ordinal concordance.
#'   \item \strong{L1 difference in weights}: Measures sensitivity of the weight
#'     vector to resampling fluctuations.
#' }
#'
#' This procedure is particularly useful when selecting among alternative
#' machine-learning weighting methods by providing empirical evidence of which
#' method yields the most stable and reliable DL2 composite indicator.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' Z <- matrix(runif(200), nrow = 20)
#'
#' cv <- dl2_cv_resampling(Z, weight_method = "rf", B = 50)
#' summary(cv$rho_spearman)
#' summary(cv$L1_w)
#' }
#'
#' @export
dl2_cv_resampling <- function(
    Z,
    weight_method = c("rf", "mars", "nn"),
    B            = 100,
    prop         = 0.8,
    p            = 2,
    ntree        = 500,
    max_iter     = 50,
    lambda       = 0.2,
    n_stable     = 3,
    seed         = NULL,
    verbose      = TRUE
) {
  weight_method <- match.arg(weight_method)
  Z <- as.matrix(Z)
  n <- nrow(Z)

  if (!is.null(seed)) {
    set.seed(seed)
  }

  if (verbose) {
    cat("Fitting baseline DL2 with method:", weight_method, "\n")
  }

  ## 1) Solución base en la muestra completa
  res_base <- ci_ml_fixedpoint(
    Z            = Z,
    p            = p,
    ntree        = ntree,
    max_iter     = max_iter,
    lambda       = lambda,
    n_stable     = n_stable,
    seed         = seed,
    verbose      = FALSE,
    weight_method = weight_method
  )

  ci_base <- res_base$ci_final
  w_base  <- res_base$w_final

  ## 2) Estructuras para guardar resultados de validación
  rho_spearman <- numeric(B)
  rho_kendall  <- numeric(B)
  L1_w         <- numeric(B)
  n_sub        <- floor(prop * n)

  if (verbose) {
    cat("Starting resampling validation with", B, "replications\n")
  }

  for (b in seq_len(B)) {
    if (verbose && (b %% 10 == 0)) {
      cat("  Replication", b, "of", B, "\n")
    }

    ## 2.1) Submuestra (sin reemplazo) o bootstrap (con reemplazo)
    idx_b <- sample.int(n, size = n_sub, replace = TRUE)  # bootstrap
    Z_b   <- Z[idx_b, , drop = FALSE]

    ## 2.2) Recalcular DL2 en la submuestra
    res_b <- ci_ml_fixedpoint(
      Z            = Z_b,
      p            = p,
      ntree        = ntree,
      max_iter     = max_iter,
      lambda       = lambda,
      n_stable     = n_stable,
      seed         = if (is.null(seed)) NULL else seed + b,
      verbose      = FALSE,
      weight_method = weight_method
    )

    ci_b <- res_b$ci_final
    w_b  <- res_b$w_final

    ## 2.3) Comparar índice en las mismas unidades
    #      (las unidades de la submuestra se indexan por idx_b)
    ci_base_sub <- ci_base[idx_b]

    rho_spearman[b] <- suppressWarnings(
      cor(ci_base_sub, ci_b, method = "spearman")
    )
    rho_kendall[b]  <- suppressWarnings(
      cor(ci_base_sub, ci_b, method = "kendall")
    )

    ## 2.4) Diferencia entre pesos base y pesos reestimados
    L1_w[b] <- sum(abs(w_b - w_base))
  }

  if (verbose) {
    cat("Resampling validation finished.\n")
  }

  list(
    method          = weight_method,
    rho_spearman    = rho_spearman,
    rho_kendall     = rho_kendall,
    L1_w            = L1_w,
    summary_spearman = summary(rho_spearman),
    summary_kendall  = summary(rho_kendall),
    summary_L1_w     = summary(L1_w),
    res_base        = res_base
  )
}
