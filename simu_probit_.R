library(torch);library(glmnet) ;library(ncvreg);library(parallel)
source('funtion_ncv.R')
n <- 200      # number of training samples
p <- 320        # number of features
b_true <- c(1, -1, .5, rep(0,p-3))   # true coefficient vector
n_test = 5000          # number of test samples
rho <- 0.              # feature correlation
Sigma <- outer(1:p, 1:p, function(i, j)rho^abs(i - j)) ; LL = chol(Sigma) 


pconfC_out = scad_out = mcp_out = ell1_out = glmnet_out = oldscad_out = oldmcp_out = list()
for (ss in 1:50) {
  x_train <- matrix(rnorm(n * p), ncol = p)%*% LL  # training features
  eta <- x_train %*% b_true              # linear predictor
  # confidence scores for positive samples
  r_all <- pnorm(eta) 
  y_train <- rbinom(n, 1, r_all)            # training labels
  r_train <- r_all[y_train == 1]            # confidence scores for positive samples only
  x_train_p <- x_train[y_train == 1, ]      # training features for positive samples only
  
  x_test <- matrix(rnorm(n_test * p), ncol = p)      # test features
  y_test <- ifelse(x_test %*% b_true  >= 0, 1, 0) # + rnorm(n_test) # test labels
  
  # fit simple logistic loss pconfClassification 
  res_naive <- pconfClassification(  num_epochs = 100, lr = 0.01, x_train_p = x_train_p,  
                                     x_test = x_test,   y_test = y_test,  r = r_train )
  bhat_naive <- res_naive$params$linear.weight
  pconfC_out[[ss]] = c(res_naive$accuracy,'est'= sum( (b_true - bhat_naive)^2), variable_select(b_true, bhat_naive) )
  
  
  #+++ fit SCAD pconf-Classification
  cv_ncv <- pconf_cv_lambda_ncpen(x = x_train_p, r = r_train , penalty = 'SCAD', verbose = F,
                                  lambda_grid = 10^seq(-1, .01, length.out = 15) ) 
  res_SCAD <- pconf_ncpen_noy(  num_epochs = 200, lr = 0.01,penalty = 'SCAD',
                                x_train = x_train_p, x_test = x_test,   y_test = y_test,  r = r_train, 
                                lambda =  cv_ncv$best_lambda
  ) ; bhat_SCAD <-  res_SCAD$w
  scad_out[[ss]] = c(res_SCAD$accuracy, 'est'=  sum( (b_true - bhat_SCAD)^2) , variable_select(b_true, bhat_SCAD) )
  
  
  #+++  fit MCP pconf-Classification
  cv_ncv <- pconf_cv_lambda_ncpen(x = x_train_p, r = r_train,penalty = 'MCP', verbose = F, a =3,
                                  lambda_grid = 10^seq(-1, .01, length.out = 15) ) 
  res_MCP <- pconf_ncpen_noy(  num_epochs = 200, lr = 0.01,penalty = 'MCP', a = 3, 
                               x_train = x_train_p,   x_test = x_test,   y_test = y_test,
                               r = r_train, lambda =  cv_ncv$best_lambda
  )  ; bhat_MCP <-  res_MCP$w
  mcp_out[[ss]] = c(res_MCP$accuracy, 'est'=sum( (b_true - bhat_MCP)^2) , variable_select(b_true, bhat_MCP) )
  
  
  #+++  fit Lasso pconf-Classification
  cv_ncv <- pconf_cv_lambda_ncpen(x = x_train_p, r = r_train,penalty = 'L1', verbose = F,
                                  lambda_grid = 10^seq(-1, .01, length.out = 15) ) 
  res_L1 <- pconf_ncpen_noy(  num_epochs = 200, lr = 0.01,penalty = 'L1',
                              x_train = x_train_p,   x_test = x_test,    y_test = y_test,
                              r = r_train, lambda =  cv_ncv$best_lambda
  ) ; bhat_L1 <-  res_L1$w
  ell1_out[[ss]] = c(res_L1$accuracy, 'est'=sum( (b_true - bhat_L1)^2), variable_select(b_true, bhat_L1) )
  
  # fit classical scad
  cvncvreg <- cv.ncvreg(X = x_train,y = y_train,penalty = 'SCAD', nfolds = 5,family = "binomial", max.iter = 50000)
  scad_b = cvncvreg$fit$beta[,cvncvreg$min][-1]
  y_pred_scad <- ifelse(sigmoid_stable(x_test %*% scad_b)  >= 0.5, 1, 0)
  oldscad_out[[ss]] = c( mean(y_pred_scad == y_test), 'est'=sum( (b_true - scad_b)^2) , variable_select(b_true, scad_b) )
  
  # fit classical MCP
  cvncvreg <- cv.ncvreg(X = x_train,y = y_train,penalty = 'MCP', nfolds = 5,family = "binomial", max.iter = 50000)
  mcp_b = cvncvreg$fit$beta[,cvncvreg$min][-1]
  y_pred_mcp <- ifelse(sigmoid_stable(x_test %*% mcp_b)  >= 0.5, 1, 0)
  oldmcp_out[[ss]] = c( mean(y_pred_mcp == y_test), 'est'=sum( (b_true - mcp_b)^2) , variable_select(b_true, mcp_b) )
  
  
  
  #+++  fit glmnet logistic regression for whole data
  cvglmnet <- cv.glmnet(x = x_train, y = y_train, family = "binomial", nfolds = 5)
  bhat_glmnet <- as.vector(coef(cvglmnet, s = "lambda.min"))[-1]
  y_pred_glmnet <- ifelse(sigmoid_stable(x_test %*% bhat_glmnet)  >= 0.5, 1, 0)
  glmnet_out[[ss]] = c(mean(y_pred_glmnet == as.numeric(y_test)), 'est'=sum( (b_true - bhat_glmnet)^2), variable_select(b_true, bhat_glmnet) )
  cat(ss,';')
}
rm(x_test,x_train,x_train_p,cvglmnet)
save.image('sim_probit_n200p320_.rda')
