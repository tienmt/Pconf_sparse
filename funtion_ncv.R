

variable_select <- function(beta_hat, beta_true) {
  # Selected variables
  selected <- beta_hat != 0
  
  # True active variables
  active <- beta_true != 0
  
  # True positives: selected AND truly active
  TP <- sum(selected & active)
  
  # False positives: selected but not active
  FP <- sum(selected & !active)
  
  # False negatives: active but not selected
  FN <- sum(!selected & active)
  
  # TPR: TP / (TP + FN)
  TPR <- if ((TP + FN) == 0) NA else TP / (TP + FN)
  
  # FDR: FP / (TP + FP)
  FDR <- if ((TP + FP) == 0) NA else FP / (TP + FP)
  
  return(c(TPR = round( TPR , 2), FDR = round(FDR,2) ))
}



# Define simple linear model
LinearNetwork <- nn_module(
  initialize = function(input_size, output_size) {
    self$linear <- nn_linear(input_size, output_size,bias = FALSE)
  },
  forward = function(x) {   self$linear(x) }
)

# Logistic loss: ℓ(z) = log(1 + exp(-z))
logistic_loss <- function(z)  torch_log1p(torch_exp(-z))


# Accuracy function
getAccuracy <- function(x_test, y_test, model) {
  with_no_grad({
    logits <- model(x_test)
    preds <- as.numeric(logits > 0)
    mean(preds == as.numeric(y_test))
  })
}

# Pconf classifier with correct loss
pconfClassification <- function(num_epochs, lr, x_train_p, x_test, y_test, r) {
  # Convert to torch tensors
  x_train <- torch_tensor(x_train_p, dtype = torch_float())
  r_tensor <- torch_tensor(r, dtype = torch_float())
  x_test_t <- torch_tensor(x_test, dtype = torch_float())
  y_test_t <- torch_tensor(y_test, dtype = torch_float())
  
  # Model & optimizer
  model <- LinearNetwork(input_size = ncol(x_train_p), output_size = 1)
  optimizer <- optim_sgd(model$parameters, lr = lr)
  
  loss_history <- numeric(num_epochs)
  
  # Training
  for (epoch in 1:num_epochs) {
    optimizer$zero_grad()
    g <- model(x_train)
    
    # Correct Pconf loss:
    # mean_i [ ℓ(g(x_i)) + ((1 - r_i)/r_i) * ℓ(-g(x_i)) ]
    loss <- logistic_loss(g) + ((1 - r_tensor) / r_tensor) * logistic_loss(-g)
    loss <- torch_mean(loss)
    
    loss$backward()
    optimizer$step()
    
    loss_history[epoch] <- loss$item()
  }
  # Parameters & accuracy
  params <- lapply(model$parameters, function(p) as.numeric(p$detach()$to(device = "cpu")))
  accuracy <- getAccuracy(x_test_t, y_test_t, model)
  
  list(params = params, accuracy = accuracy, loss_history =loss_history)
}



# stable logistic loss l(z) = log(1 + exp(-z))
logistic_loss_stable <- function(z) {
  out <- numeric(length(z))
  pos <- z >= 0
  out[pos]  <- log1p(exp(-z[pos]))              # z >= 0
  out[!pos] <- -z[!pos] + log1p(exp(z[!pos]))   # z < 0  -> use -z + log1p(exp(z))
  out
}

sigmoid_stable <- function(z) {
  s <- numeric(length(z))
  pos <- z >= 0
  s[pos] <- 1 / (1 + exp(-z[pos]))
  s[!pos] <- exp(z[!pos]) / (1 + exp(z[!pos]))
  s
}

# soft-threshold (proximal operator for L1)
soft_threshold <- function(w, thresh) {
  sign(w) * pmax(0, abs(w) - thresh)
}
prox_L1 <- function(w, t_lambda)  soft_threshold(w, t_lambda)


prox_SCAD <- function(w, lr, lambda, a = 3.7) {
  z <- w
  out <- numeric(length(z))
  
  # Region 1: |z| <= lambda + lr*lambda
  idx1 <- abs(z) <= (lambda + lr * lambda)
  out[idx1] <- soft_threshold(z[idx1], lr * lambda)
  
  # Region 2: lambda + lr*lambda < |z| <= a*lambda
  idx2 <- (abs(z) > (lambda + lr * lambda)) & (abs(z) <= a * lambda)
  out[idx2] <- ((a - 1) * z[idx2] - sign(z[idx2]) * a * lr * lambda) / (a - 1 - lr)
  
  # Region 3: |z| > a*lambda
  idx3 <- abs(z) > a * lambda
  out[idx3] <- z[idx3]
  
  out
}

prox_MCP <- function(w, lr, lambda, a = 3) {
  # Correct MCP proximal operator
  # Requires lr < a for denominator (1 - lr/a) > 0
  if (lr >= a) {
    stop(sprintf("lr (=%g) must be < a (=%g) for MCP proximal to be valid.", lr, a))
  }
  z <- w
  out <- numeric(length(z))
  t <- lr
  idx1 <- abs(z) <= a * lambda
  # soft-threshold first
  S <- soft_threshold(z[idx1], t * lambda)
  denom <- (1 - t / a)
  out[idx1] <- S / denom
  idx2 <- abs(z) > a * lambda
  out[idx2] <- z[idx2]
  out
}

prox_penalty <- function(w, lr, lambda, penalty = c("L1","SCAD","MCP"), a = 3.7) {
  penalty <- match.arg(penalty)
  if (lambda == 0) return(w)
  
  switch(penalty,
         L1   = prox_L1(w, lr * lambda),
         SCAD = prox_SCAD(w, lr, lambda, a),
         MCP  = prox_MCP(w, lr, lambda, a)
  )
}

pconf_ncpen <- function(
    num_epochs, lr,
    x_train, r_train,
    lambda = 0.0,
    penalty = c("L1", "SCAD", "MCP"),
    a = 3.7,
    init_scale = 0.01,
    verbose = FALSE
) {
  penalty <- match.arg(penalty)
  
  if (!is.matrix(x_train)) x_train <- as.matrix(x_train)
  
  n <- nrow(x_train)
  p <- ncol(x_train)
  
  alpha <- (1 - r_train) / r_train
  
  # init
  set.seed(1)
  w <- rnorm(p, 0, init_scale)
  b <- 0
  
  for (epoch in 1:num_epochs) {
    g <- as.vector(x_train %*% w )
    
    s_g     <- sigmoid_stable(g)
    s_neg_g <- sigmoid_stable(-g)
    
    # derivative of logistic Pconf loss
    d <- -s_neg_g + alpha * s_g
    
    grad_w <- as.vector(t(x_train) %*% d) / n
    # grad_b <- mean(d)
    
    # gradient step
    w <- w - lr * grad_w
    # b <- b - lr * grad_b
    
    # SCAD/MCP/L1
    w <- prox_penalty(w, lr, lambda, penalty, a)
    
    if (verbose && epoch %% max(1, floor(num_epochs/10)) == 0) {
      loss <- mean(logistic_loss_stable(g) + alpha * logistic_loss_stable(-g))
      cat(sprintf("Epoch %4d | loss = %.6f | ||w||0 = %d\n",
                  epoch, loss, sum(w != 0)))
    }
  }
  
  list(w = w, b = b)
}



pconf_ncpen_noy <- function(
    num_epochs, lr,
    x_train, r_train,
    x_test = NULL, y_test = NULL,
    lambda = 0.0,
    penalty = c("L1", "SCAD", "MCP"),
    a = 3.7,
    init_scale = 0.01,
    tol_w = 1e-8,
    tol_loss = 1e-8,
    verbose = FALSE
) {
  penalty <- match.arg(penalty)
  
  if (!is.matrix(x_train)) x_train <- as.matrix(x_train)
  
  n <- nrow(x_train)
  p <- ncol(x_train)
  
  alpha <- (1 - r_train) / r_train
  
  # initialization
  set.seed(1)
  w <- rnorm(p, 0, init_scale)
  b <- 0
  
  prev_loss <- Inf
  
  for (epoch in 1:num_epochs) {
    
    w_old <- w
    
    g <- as.vector(x_train %*% w + b)
    s_g     <- sigmoid_stable(g)
    s_neg_g <- sigmoid_stable(-g)
    
    # gradient of Pconf logistic loss
    d <- -s_neg_g + alpha * s_g
    grad_w <- as.vector(t(x_train) %*% d) / n
    
    # gradient step
    w <- w - lr * grad_w
    
    # proximal step
    w <- prox_penalty(w, lr, lambda, penalty, a)
    
    # recompute loss after update
    g_new <- as.vector(x_train %*% w + b)
    loss <- mean(  logistic_loss_stable(g_new) +
        alpha * logistic_loss_stable(-g_new)
    )
    
    # stopping rules
    rel_w_err <- norm(w - w_old, "2") / max(1, norm(w_old, "2"))
    loss_err  <- abs(loss - prev_loss)
    
    if (verbose && epoch %% max(1, floor(num_epochs / 10)) == 0) {
      cat(sprintf(
        "Epoch %4d | loss = %.6e | rel_w = %.2e | nnz = %d\n",
        epoch, loss, rel_w_err, sum(w != 0)
      ))
    }
    
    if (rel_w_err < tol_w || loss_err < tol_loss) {
        cat(sprintf("Converged at epoch %d (rel_w = %.2e, loss_err = %.2e)\n",
          epoch, rel_w_err, loss_err  ))
      break
    }
    
    prev_loss <- loss
  }
  
  out <- list(w = w, b = b, loss = loss)
  
  if (!is.null(x_test) && !is.null(y_test)) {
    g_test <- as.vector(x_test %*% w + b)
    preds_prob <- sigmoid_stable(g_test)
    y_pred <- ifelse(preds_prob >= 0.5, 1, 0)
    out$accuracy <- mean(y_pred == as.numeric(y_test))
  }
  
  out
}



pconf_cv_lambda_ncpen <- function(
    x, r,
    lambda_grid,
    K = 5,
    num_epochs = 500,
    lr = 0.1,
    penalty = c("L1", "SCAD", "MCP"),
    a = 3.7,
    verbose = TRUE
) {
  penalty <- match.arg(penalty)
  
  if (!is.matrix(x)) x <- as.matrix(x)
  n <- nrow(x)
  
  set.seed(123)
  folds <- sample(rep(1:K, length.out = n))
  
  cv_risk <- numeric(length(lambda_grid))
  
  for (li in seq_along(lambda_grid)) {
    lambda <- lambda_grid[li]
    fold_risks <- numeric(K)
    
    if (verbose) cat(sprintf("\nTesting lambda = %g\n", lambda))
    
    for (k in 1:K) {
      idx_valid <- which(folds == k)
      idx_train <- setdiff(seq_len(n), idx_valid)
      
      x_train <- x[idx_train, , drop = FALSE]
      r_train <- r[idx_train]
      
      x_valid <- x[idx_valid, , drop = FALSE]
      r_valid <- r[idx_valid]
      
      fit <- pconf_ncpen(
        num_epochs = num_epochs,
        lr = lr,
        x_train = x_train,
        r_train = r_train,
        lambda = lambda,
        penalty = penalty,
        a = a,
        verbose = FALSE
      )
      
      g_val <- as.vector(x_valid %*% fit$w + fit$b)
      alpha_val <- (1 - r_valid) / r_valid
      
      fold_risks[k] <- mean(
        logistic_loss_stable(g_val) +
          alpha_val * logistic_loss_stable(-g_val)
      )
    }
    
    cv_risk[li] <- mean(fold_risks)
    if (verbose)
      cat(sprintf("  Mean CV risk: %.6f\n", cv_risk[li]))
  }
  
  best_lambda <- lambda_grid[which.min(cv_risk)]
  
  list(
    lambda_grid = lambda_grid,
    cv_risk = cv_risk,
    best_lambda = best_lambda,
    penalty = penalty
  )
}
