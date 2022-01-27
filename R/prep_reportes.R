#devtools::install_github("cotecora-team-2/quickcountmx")
library(cmdstanr)
library(posterior)
library(tidyverse)
library(patchwork)
library(quickcountmx)
theme_set(theme_minimal())
cb_palette <- c("#000000", "#E69F00", "#56B4E9", "#009E73",
                "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


estados <- c("MICH", "CHIH", "COL", "ZAC", "NAY")
partidos_todos <- c("AMLO", "JAMK", "RAC", "CAND_IND_01", "CAND_IND_02", "OTROS")
partidos <- c("AMLO", "JAMK", "RAC", "CAND_IND_02", "OTROS")

#sims <- read_rds("./datos/simulaciones_completo.rds")
#sims_llegadas <- sims %>% group_by(id, state_abbr) %>%
#  mutate(prop_observado = percent_rank(time)) %>%
#  ungroup


## Estratificaciones
data("conteo_2018")
conteo_15 <- read_csv("./data/conteo_15_inegi_10_sin_imputar.csv")
estratos_colapsado <- read_csv("./data/estratos_colapsados_con_chih.csv")
comp_15 <- conteo_15 %>%
  group_by(ID_ESTADO, ID_DISTRITO = ID_DTTO_FEDERAL_jul2020,
           SECCION = SECCION_jul2020) %>%
  summarise(comp_votos = mean(.fittedPC1)) %>%
  ungroup() %>%
  mutate(SECCION = stringr::str_pad(SECCION, 4, pad = "0"))
# configuraciones
estratif_conf <- crossing(
  num_gpos_marginacion = c(1,2),
  num_gpos_ln = c(1,2),
  num_gpos_votos = c(1, 2)) %>%
  mutate(estratificacion_num = row_number()) %>%
  transpose()
# computos distritales
conteo_2018_estados <- conteo_2018 %>%
  filter(state_abbr %in% estados) %>%
  group_by(state_abbr) %>%
  left_join(comp_15)
conteo_2018_estados <- conteo_2018_estados %>%
  mutate(OTROS = CNR + VN) %>%
  group_by(ID_ESTADO, ID_DISTRITO) %>%
  mutate(comp_votos = ifelse(is.na(comp_votos), mean(comp_votos, na.rm = TRUE),
                             comp_votos))
# preparar estratificaciones
marco_conf <- map_df(estratif_conf, function(x){
  num_gpos_marginacion <- x$num_gpos_marginacion
  num_gpos_ln <- x$num_gpos_ln
  num_gpos_votos <- x$num_gpos_votos
  estratificacion_num <- x$estratificacion_num
  marco <- conteo_2018_estados %>%
    group_by(state_abbr) %>%
    mutate(estratificacion_num = estratificacion_num) %>%
    mutate(indice_grupo = ntile(.fittedPC1, num_gpos_marginacion)) %>%
    mutate(indice_votos = ntile(comp_votos, num_gpos_votos)) %>%
    mutate(tam_ln = ntile(LISTA_NOMINAL_CASILLA, num_gpos_ln)) %>%
    mutate(estrato_df = interaction(ID_DISTRITO, indice_grupo, indice_votos,
                                    tam_ln))
})

## colapsar estratos chicos
marco_conf <- left_join(marco_conf, estratos_colapsado %>%
      select(state_abbr, estratificacion_num, estrato_df, estrato_df_colapsado))
marco_conf <- marco_conf %>% select(-estrato_df) %>%
  rename(estrato_df = estrato_df_colapsado)


estratif_selec <- tibble(
  state_abbr = c("CHIH", "MICH", "COL", "NAY", "ZAC"),
  estratificacion_num = c(3, 3, 6, 6, 6)
)

marco_conf_selec <- marco_conf %>% semi_join(estratif_selec) %>%
  mutate(NOMBRE_ESTADO = ifelse(state_abbr == "MICH", "MICHOACAN", NOMBRE_ESTADO))

## filtrar estado seleccionado

datos_ent <- marco_conf_selec %>% filter(NOMBRE_ESTADO == estado) %>%
  # usar estratificacion anterior por ahora
  mutate(estrato_df = as.numeric(factor(estrato_df))) %>%
  select(any_of(partidos_todos), CLAVE_CASILLA, estrato_df,
         ln = LISTA_NOMINAL_CASILLA, componente = .fittedPC1, TIPO_CASILLA.x) %>%
  mutate(in_sample = 0, y_f = 0) %>%
  mutate(ln = ifelse(ln == 0, 1200, ln)) %>%
  mutate(casilla_esp = ifelse(TIPO_CASILLA.x == "S", 1, 0)) %>%
  mutate(no_casilla = row_number()) %>%
  ungroup()
# agrupar partidos
datos_ent <- datos_ent %>%
  mutate(OTROS = OTROS + CAND_IND_01)
datos_ent$CAND_IND_01 <- NULL

sim_datos_lista <- list()
x <- datos_ent %>%
  select(ln, componente) %>%
  as.matrix()
x_f <- scale(x)
sim_datos_lista$x_f <- x_f
sim_datos_lista$N_f <- nrow(x_f)
sim_datos_lista$n_covariates_f <- ncol(x_f)
sim_datos_lista$n_strata_f <- datos_ent$estrato_df %>% unique %>% length()
sim_datos_lista$n_f <- datos_ent$ln
sim_datos_lista$stratum_f <- datos_ent$estrato_df
jsonlite::write_json(sim_datos_lista, "data/datos_prueba.json")

sim_datos <- jsonlite::read_json("./data/datos_prueba.json", simplifyVector = TRUE)
parametros <- jsonlite::read_json("./data/prior_data.json", simplifyVector = TRUE)
