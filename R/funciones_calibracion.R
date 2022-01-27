obtener_muestra <- function(sims_casillas, num_draw, frac= 0.04, seed = NA){
  sims_1 <- sims_casillas %>% filter(.draw == num_draw)
  muestra_1 <- quickcountmx::select_sample_prop(sims_1, stratum = estrato_df, frac = frac,
                                                seed = seed)
  in_sample_na <- sims_1 %>% select(no_casilla) %>%
    left_join(muestra_1 %>% select(no_casilla, estrato_df),
              by = "no_casilla") %>% pull(estrato_df)
  in_sample <- ifelse(is.na(in_sample_na), 0, 1) %>% as.integer()
  list(y = muestra_1$y_f, N = length(muestra_1$y_f), stratum = muestra_1$estrato_df,
       n = muestra_1$ln, x = x_f[muestra_1$no_casilla, , drop = FALSE],
       in_sample = in_sample, y_f = sims_1$y_f)
}

obtener_muestra_marco <- function(marco_tbl, frac = 0.04, seed = NA,
                                  prop_obs = 1.0, sims_llegadas = NULL,
                                  id_selec=NULL, partidos = partidos){
  muestra_1 <- quickcountmx::select_sample_prop(marco_tbl,
                                                stratum = estrato_df,
                                                frac = frac,
                                                seed = seed)
  if(prop_obs < 1.0){
    if(is.null(id_selec)){
      id_selec <- sample(1:500, 1)
    }
    muestra_1 <- muestra_1 %>% left_join(sims_llegadas %>% filter(id == id_selec),
                             by = c("CLAVE_CASILLA", "state_abbr"))
    muestra_1 <- muestra_1 %>%
      ungroup %>%
      mutate(prop_obs_muestra = percent_rank(time))
    muestra_1 <- filter(muestra_1, prop_obs_muestra <= prop_obs)
  }
  in_sample_na <- marco_tbl %>% select(no_casilla) %>%
    left_join(muestra_1 %>% select(no_casilla, estrato_df),
              by = "no_casilla") %>%
    pull(estrato_df)
  in_sample <- ifelse(is.na(in_sample_na), 0, 1)
  votos <- muestra_1 %>% select(any_of(partidos))
  list(y = muestra_1$y_f, N = length(muestra_1$y_f),
       in_sample = in_sample,
       stratum = muestra_1$estrato_df,
       n = muestra_1$ln, x = x_f[muestra_1$no_casilla, , drop = FALSE],
       votos = votos)
}

ajustar_diagnosticos <- function(sims_casillas, rep, frac = 0.04, modelo, datos, params,
                                 adapt_delta = 0.9, max_treedepth = 10,
                                 iter_sampling=2000, iter_warmup = 2000){

  datos_muestra <- obtener_muestra(sims_casillas, rep, frac = frac, seed = NA)
  datos_1 <- c(datos, datos_muestra)
  print(rep)
  ajuste <- modelo$sample(data = datos_1,
                          seed = 2210,
                          iter_sampling = iter_sampling, iter_warmup = iter_warmup,
                          parallel_chains = 4,
                          adapt_delta = adapt_delta,
                          max_treedepth = max_treedepth,
                          refresh = 0,
                          show_messages = FALSE)


  suppressMessages(diagnostico <- ajuste$cmdstan_diagnose())
  suppressMessages(resumen <- ajuste$summary())

  # diagnosticar parámetros
  sims_tbl <- ajuste$draws(names(params)) %>% as_draws_df() %>% as_tibble()
  sbc_tbl <- sbc_rank(params, sims_tbl)
  tibble(rep = rep, params = list(params), sbc_rank = list(sbc_tbl),
         resumen = list(resumen), diagnosticos = list(diagnostico))
}

sbc_rank <- function(params_tbl, sims_tbl){
  params_nom <- names(params_tbl)
  sims_tbl_larga <- sims_tbl %>%
    filter((row_number() %% 10) == 0) %>% # adelgazar la cadena
    pivot_longer(cols = any_of(params_nom), names_to = "parametro", values_to = "valor")
  params_tbl_larga <- params_tbl %>%
    pivot_longer(cols = any_of(params_nom), names_to = "parametro", values_to = "valor_real")
  sbc_tbl <- sims_tbl_larga %>%
    left_join(params_tbl_larga, by = "parametro") %>%
    group_by(parametro) %>%
    summarise(sbc_rank = mean(valor_real < valor))
  sbc_tbl %>% pivot_wider( names_from = "parametro", values_from ="sbc_rank")
}
