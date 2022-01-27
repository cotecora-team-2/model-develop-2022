functions{
real normal_lb_rng(real mu, real sigma, real lb) {
  real p_lb = normal_cdf(lb | mu, sigma);
  real u = uniform_rng(p_lb, 1.0);
  real y = mu + sigma * Phi(u);
  return y;
}
}

data {

  int N_f; // total number of stations
  int n_strata_f;
  int n_covariates_f;
  int p; // number of parties
  //int<lower=0> y_f[N_f] ; // observed vote counts
  vector[N_f] n_f; // nominal counts
  int stratum_f[N_f];
  matrix[N_f, n_covariates_f] x_f;

  // conf
  vector[2] beta_0_param;
  real sigma_param;
  vector[2] kappa_param;
  real sigma_coefs;
}

transformed data {
  real<lower=0> total_nominal;

  total_nominal = 0;
  for(i in 1:N_f){
    if(n_f[i] < 1200){
      total_nominal += n_f[i];
    }
  }

}

parameters {

}

transformed parameters {

}

model {

}

generated quantities {
  row_vector[p] beta_0;
  real beta_0_part;
  matrix[n_covariates_f, p] beta;
  vector[n_covariates_f] beta_part;
  row_vector<lower=0>[p] kappa_0;
  vector<lower=0>[p] sigma;
  vector<lower=0>[p] sigma_kappa;
  real<lower=0> sigma_part;
  matrix[n_strata_f, p] beta_st_raw;
  matrix[n_strata_f, p] kappa_st_raw;
  vector[n_strata_f] beta_st_part_raw;
  vector<lower=0>[n_strata_f] kappa_part;
  // transformed
  //matrix[N_f,p] pred;
  vector[p] theta[N_f];
  vector[N_f] alpha_bn[p];
  matrix[n_strata_f,p] beta_st;
  vector[N_f] theta_part;
  vector[N_f] alpha_bn_part;
  vector[n_strata_f] beta_st_part;
  //vector[N_f] pred_part;
  matrix<lower=0>[n_strata_f, p] kappa;
  vector[p] y_out;
  // generated
  real prop_votos[p];
  vector[p] theta_f;
  real alpha_bn_f_part;
  vector[p] alpha_bn_f;
  vector[p] pred_f;
  real pred_f_part;
  real theta_f_total[N_f];
  real total_cnt;
  real participacion;
  real total_est[N_f];
  // simulate parameters
  for(k in 1:p){
    beta_0[k] = normal_rng(beta_0_param[1], beta_0_param[2]);
  }
  beta_0_part = normal_rng(beta_0_param[1], beta_0_param[2]);
  for(k in 1:p){
    for(i in 1:n_covariates_f){
      beta[i,k] = normal_rng(0, 1);
    }
  }
  for(k in 1:p){
    for(i in 1:n_strata_f){
        kappa_st_raw[i, k] = normal_rng(0, 1);
        beta_st_raw[i,k] = normal_rng(0, 1);
    }
  }
  for(i in 1:n_covariates_f){
    beta_part[i] = normal_rng(0, 1);
  }
  for(k in 1:p){
      kappa_0[k] = normal_lb_rng(2, 1, 0);
  }
  for(i in 1:n_strata_f){
    beta_st_part_raw[i] = normal_rng(0, 1);
    kappa_part[i] = gamma_rng(kappa_param[1], kappa_param[2]);
  }
  for(k in 1:p){
    sigma[k] = fabs(normal_rng(0, sigma_param));
    sigma_kappa[k] = fabs(normal_rng(0, 0.25));
  }
  sigma_part = fabs(normal_rng(0, sigma_param));
  // transform derived parameters
  kappa = exp(rep_matrix(kappa_0, n_strata_f) + diag_post_multiply(kappa_st_raw, sigma_kappa));

  beta_st_part = beta_0_part + beta_st_part_raw * sigma_part;
  beta_st = rep_matrix(beta_0, n_strata_f) + diag_post_multiply(beta_st_raw, sigma);

  total_cnt = 0;
    for(k in 1:p){
      y_out[k] = 0.0;
    }
  for(i in 1:N_f){
      pred_f_part = sigma_coefs * dot_product(x_f[i,], beta_part);
      theta_f_total[i] = inv_logit(beta_st_part[stratum_f[i]] + pred_f_part);
      alpha_bn_f_part =  n_f[i] * theta_f_total[i];
      total_est[i] = neg_binomial_2_rng(alpha_bn_f_part , alpha_bn_f_part/kappa_part[stratum_f[i]]);
      total_cnt += total_est[i];
//party vote

      pred_f = sigma_coefs * to_vector(x_f[i,] * beta);
      theta_f = softmax(to_vector(beta_st[stratum_f[i],]) + pred_f);
      alpha_bn_f =  n_f[i] * theta_f * theta_f_total[i];
    for(k in 1:p){
      y_out[k] += neg_binomial_2_rng(alpha_bn_f[k], alpha_bn_f[k] / kappa[stratum_f[i], k]);
    }
  }
  for(k in 1:p){
    prop_votos[k] = y_out[k] / total_cnt;
  }
  participacion = total_cnt / total_nominal;
}

