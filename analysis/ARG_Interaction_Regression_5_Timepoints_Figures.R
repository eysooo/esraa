#!/usr/bin/env Rscript

# ============================================================================

message("Run this complete file with RStudio's Source button; do not paste it into the console in sections.")
# ARG trial: longitudinal interaction regression + repeated-measures ANOVA
# Updated for Baseline, Month 3, Month 6, Month 9, and Month 12.
#
# Primary model equation retained from the attached script:
#   log_y ~ timepoint * intervention + (1 | code_number)
#
# Outputs:
#   1. Type III mixed-model omnibus tests with BH-FDR correction
#   2. Estimated marginal means and Holm-adjusted pairwise comparisons
#   3. Greenhouse-Geisser repeated-measures ANOVA sensitivity analysis
#   4. One publication-quality figure per available outcome
#      (600-dpi PNG + 600-dpi TIFF + one combined vector PDF)
#   5. Model summaries, QC tables, and session information
# ============================================================================

required_packages <- c(
  "readxl", "janitor", "dplyr", "tidyr", "stringr", "tibble",
  "lme4", "lmerTest", "emmeans", "afex", "ggplot2", "scales"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Install the missing packages first with:\ninstall.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(readxl)
  library(janitor)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(tibble)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(afex)
  library(ggplot2)
  library(scales)
})

# Sum-to-zero contrasts are required for interpretable Type III tests.
options(contrasts = c("contr.sum", "contr.poly"))

# -----------------------------------------------------------------------------
# 1. File path
# -----------------------------------------------------------------------------
# The first path is the usual macOS path. The second preserves the extra space
# before the slash in the path supplied in the message, in case that is the
# literal folder name.
data_candidates <- path.expand(c(
  "~/Desktop/Qatar University/PhD Project/project documents/new results/ARG FULL DATA.xlsx",
  "~/Desktop/Qatar University/PhD Project/project documents/new results /ARG FULL DATA.xlsx"
))

existing_paths <- data_candidates[file.exists(data_candidates)]

if (length(existing_paths) == 0) {
  stop(
    "ARG FULL DATA.xlsx was not found. Checked:\n",
    paste0("- ", data_candidates, collapse = "\n"),
    call. = FALSE
  )
}

data_path <- existing_paths[[1]]
ARG_FULL_DATA <- read_excel(data_path)

message("Reading data from: ", data_path)

# -----------------------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------------------
numeric_clean <- function(x) {
  suppressWarnings(
    as.numeric(str_replace_all(str_trim(as.character(x)), fixed(","), "."))
  )
}

format_p <- function(p) {
  if (length(p) == 0 || is.na(p)) return("NA")
  if (p < 0.001) return(formatC(p, format = "e", digits = 2))
  sprintf("%.3f", p)
}

safe_filename <- function(x) {
  x |>
    str_replace_all("[^A-Za-z0-9_-]+", "_") |>
    str_replace_all("_+", "_") |>
    str_remove("^_") |>
    str_remove("_$")
}

first_matching_row <- function(x, candidates) {
  hit <- candidates[candidates %in% rownames(x)]
  if (length(hit) == 0) return(NA_character_)
  hit[[1]]
}

first_matching_column <- function(x, pattern) {
  hit <- grep(pattern, names(x), ignore.case = TRUE, value = TRUE)
  if (length(hit) == 0) return(NA_character_)
  hit[[1]]
}

bt <- function(x, constant) exp(x) - constant

# -----------------------------------------------------------------------------
# 3. Clean and recode the workbook
# -----------------------------------------------------------------------------
data <- ARG_FULL_DATA |>
  clean_names()

# Drop fully empty spreadsheet columns without altering the raw imported object.
data <- data[, colSums(!is.na(data)) > 0, drop = FALSE]

# Handle possible transliteration variants produced by clean_names().
rename_first_alias <- function(df, canonical, aliases) {
  if (canonical %in% names(df)) return(df)
  available <- aliases[aliases %in% names(df)]
  if (length(available) > 0) {
    names(df)[names(df) == available[[1]]] <- canonical
  }
  df
}

data <- rename_first_alias(data, "g_gt", c("gamma_gt", "y_gt", "gt"))
data <- rename_first_alias(data, "tnf_a", c("tnf_alpha", "tnf_alfa"))

required_columns <- c("code_number", "intervention", "timepoint")
missing_key_columns <- setdiff(required_columns, names(data))

if (length(missing_key_columns) > 0) {
  stop(
    "Required column(s) were not found after clean_names(): ",
    paste(missing_key_columns, collapse = ", "),
    call. = FALSE
  )
}

time_codes <- 0:4
time_labels <- c("Baseline", "Month 3", "Month 6", "Month 9", "Month 12")

data <- data |>
  mutate(
    code_number = factor(numeric_clean(code_number)),
    intervention_code = numeric_clean(intervention),
    time_code = numeric_clean(timepoint),
    intervention = factor(
      intervention_code,
      levels = c(1, 2),
      labels = c("Placebo", "Synbiotic")
    ),
    timepoint = factor(
      time_code,
      levels = time_codes,
      labels = time_labels,
      ordered = FALSE
    )
  ) |>
  filter(!is.na(code_number), !is.na(intervention), !is.na(timepoint))

duplicate_visits <- data |>
  count(code_number, timepoint, name = "n") |>
  filter(n > 1)

if (nrow(duplicate_visits) > 0) {
  stop(
    "Duplicate participant-by-timepoint rows were detected. Resolve them before modelling.",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# 4. Outcomes and manuscript labels
# -----------------------------------------------------------------------------
# Clinical outcomes use all five available visits. Cytokines are also included;
# they will automatically use only the visits that contain sufficient data.
outcome_labels <- c(
  bmi = "BMI (kg/m²)",
  homa_ir = "HOMA-IR",
  fasting_glucose = "Fasting glucose (mmol/L)",
  insulin = "Fasting insulin (µU/mL)",
  tc = "Total cholesterol (mmol/L)",
  tg = "Triglycerides (mmol/L)",
  hdl_c = "HDL-C (mmol/L)",
  ldl_c = "LDL-C (mmol/L)",
  urea = "Urea (mmol/L)",
  crea = "Creatinine (µmol/L)",
  t_bil_v = "Total bilirubin (µmol/L)",
  d_bil = "Direct bilirubin (µmol/L)",
  tp = "Total protein (g/L)",
  alb = "Albumin (g/L)",
  alt = "ALT (U/L)",
  ast = "AST (U/L)",
  ldh = "LDH (U/L)",
  g_gt = "γ-GT (U/L)",
  il_1_beta = "IL-1β (pg/mL)",
  il1_ra = "IL-1RA (pg/mL)",
  il_10 = "IL-10 (pg/mL)",
  il_22 = "IL-22 (pg/mL)",
  mcp_1_ccl2 = "MCP-1/CCL2 (pg/mL)",
  tnf_a = "TNF-α (pg/mL)",
  il_6 = "IL-6 (pg/mL)",
  il_8 = "IL-8 (pg/mL)"
)

outcome_families <- c(
  bmi = "Anthropometric",
  homa_ir = "Glycaemic", fasting_glucose = "Glycaemic", insulin = "Glycaemic",
  tc = "Lipid", tg = "Lipid", hdl_c = "Lipid", ldl_c = "Lipid",
  urea = "Renal", crea = "Renal",
  t_bil_v = "Hepatic", d_bil = "Hepatic", tp = "Hepatic", alb = "Hepatic",
  alt = "Hepatic", ast = "Hepatic", ldh = "Hepatic", g_gt = "Hepatic",
  il_1_beta = "Inflammatory", il1_ra = "Inflammatory",
  il_10 = "Inflammatory", il_22 = "Inflammatory",
  mcp_1_ccl2 = "Inflammatory", tnf_a = "Inflammatory",
  il_6 = "Inflammatory", il_8 = "Inflammatory"
)

available_outcomes <- names(outcome_labels)[names(outcome_labels) %in% names(data)]
missing_outcomes <- setdiff(names(outcome_labels), available_outcomes)

if (length(available_outcomes) == 0) {
  stop("None of the requested outcomes were found in the workbook.", call. = FALSE)
}

message("Available outcomes to analyse: ", paste(available_outcomes, collapse = ", "))

# -----------------------------------------------------------------------------
# 5. Output folders
# -----------------------------------------------------------------------------
run_tag <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path(dirname(data_path), paste0("ARG_longitudinal_results_", run_tag))

csv_dir <- file.path(out_dir, "csv")
summary_dir <- file.path(out_dir, "model_summaries")
figure_png_dir <- file.path(out_dir, "figures_png_600dpi")
figure_tiff_dir <- file.path(out_dir, "figures_tiff_600dpi")

dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_png_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_tiff_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 6. Primary mixed models
# -----------------------------------------------------------------------------
minimum_per_group_visit <- 5

model_store <- list()
omnibus_rows <- list()
emmeans_rows <- list()
pairwise_time_rows <- list()
pairwise_group_rows <- list()
inclusion_rows <- list()

for (outcome in available_outcomes) {
  dat_i <- data |>
    select(code_number, intervention, timepoint, all_of(outcome)) |>
    rename(y_raw = all_of(outcome)) |>
    mutate(y_raw = numeric_clean(y_raw)) |>
    drop_na()

  cell_counts <- dat_i |>
    count(timepoint, intervention, name = "n") |>
    complete(timepoint, intervention, fill = list(n = 0))

  eligible_times <- cell_counts |>
    group_by(timepoint) |>
    summarise(minimum_n = min(n), .groups = "drop") |>
    filter(minimum_n >= minimum_per_group_visit) |>
    pull(timepoint) |>
    as.character()

  dat_i <- dat_i |>
    filter(as.character(timepoint) %in% eligible_times) |>
    droplevels()

  reason <- NULL
  if (!"Baseline" %in% levels(dat_i$timepoint)) {
    reason <- "Baseline was unavailable after the per-cell sample-size rule"
  } else if (nlevels(dat_i$timepoint) < 2) {
    reason <- "Fewer than two eligible timepoints"
  } else if (nlevels(dat_i$intervention) < 2) {
    reason <- "Fewer than two intervention groups"
  } else if (nrow(dat_i) < 10) {
    reason <- "Insufficient complete observations"
  }

  if (!is.null(reason)) {
    inclusion_rows[[outcome]] <- tibble(
      outcome = outcome,
      outcome_label = unname(outcome_labels[[outcome]]),
      family = unname(outcome_families[[outcome]]),
      included = FALSE,
      reason = reason
    )
    next
  }

  minimum_y <- min(dat_i$y_raw, na.rm = TRUE)
  transform_constant <- if (minimum_y <= 0) abs(minimum_y) + 0.1 else 0
  dat_i <- dat_i |>
    mutate(log_y = log(y_raw + transform_constant))

  model_warnings <- character(0)

  # DO NOT CHANGE: this is the equation retained from the attached script.
  fit <- tryCatch(
    withCallingHandlers(
      lmerTest::lmer(
        log_y ~ timepoint * intervention + (1 | code_number),
        data = dat_i,
        REML = TRUE,
        control = lmerControl(
          optimizer = "bobyqa",
          optCtrl = list(maxfun = 200000)
        )
      ),
      warning = function(w) {
        model_warnings <<- c(model_warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    inclusion_rows[[outcome]] <- tibble(
      outcome = outcome,
      outcome_label = unname(outcome_labels[[outcome]]),
      family = unname(outcome_families[[outcome]]),
      included = FALSE,
      reason = paste("Model error:", conditionMessage(fit))
    )
    next
  }

  anova_table <- lmerTest::anova(fit, type = 3, ddf = "Satterthwaite")
  p_column <- first_matching_column(as.data.frame(anova_table), "^Pr\\(")

  time_row <- first_matching_row(anova_table, "timepoint")
  group_row <- first_matching_row(anova_table, "intervention")
  interaction_row <- first_matching_row(
    anova_table,
    c("timepoint:intervention", "intervention:timepoint")
  )

  p_time <- if (!is.na(time_row) && !is.na(p_column)) {
    as.numeric(anova_table[time_row, p_column])
  } else NA_real_
  p_group <- if (!is.na(group_row) && !is.na(p_column)) {
    as.numeric(anova_table[group_row, p_column])
  } else NA_real_
  p_interaction <- if (!is.na(interaction_row) && !is.na(p_column)) {
    as.numeric(anova_table[interaction_row, p_column])
  } else NA_real_

  emm_grid <- emmeans::emmeans(fit, ~ timepoint * intervention)
  emm_full <- as.data.frame(summary(emm_grid, infer = c(TRUE, TRUE), level = 0.95)) |>
    mutate(
      outcome = outcome,
      outcome_label = unname(outcome_labels[[outcome]]),
      family = unname(outcome_families[[outcome]]),
      transform_constant = transform_constant,
      emmean_backtransformed = bt(emmean, transform_constant),
      lower_95_backtransformed = bt(lower.CL, transform_constant),
      upper_95_backtransformed = bt(upper.CL, transform_constant)
    ) |>
    left_join(
      dat_i |> count(timepoint, intervention, name = "n_cell"),
      by = c("timepoint", "intervention")
    )

  pair_time <- as.data.frame(
    summary(
      pairs(emmeans::emmeans(fit, ~ timepoint | intervention), adjust = "holm")
    )
  ) |>
    mutate(
      outcome = outcome,
      outcome_label = unname(outcome_labels[[outcome]]),
      family = unname(outcome_families[[outcome]])
    )

  pair_group <- as.data.frame(
    summary(
      pairs(emmeans::emmeans(fit, ~ intervention | timepoint), adjust = "holm")
    )
  ) |>
    mutate(
      outcome = outcome,
      outcome_label = unname(outcome_labels[[outcome]]),
      family = unname(outcome_families[[outcome]])
    )

  model_store[[outcome]] <- list(
    fit = fit,
    anova = anova_table,
    data = dat_i,
    emmeans = emm_full,
    constant = transform_constant,
    warnings = unique(model_warnings)
  )

  omnibus_rows[[outcome]] <- tibble(
    outcome = outcome,
    outcome_label = unname(outcome_labels[[outcome]]),
    family = unname(outcome_families[[outcome]]),
    visits = paste(levels(dat_i$timepoint), collapse = ", "),
    n_observations = nrow(dat_i),
    n_participants = n_distinct(dat_i$code_number),
    transform = ifelse(transform_constant == 0, "log", "shifted log"),
    transform_constant = transform_constant,
    singular_fit = lme4::isSingular(fit, tol = 1e-4),
    timepoint_p = p_time,
    intervention_p = p_group,
    interaction_p = p_interaction,
    model_warnings = paste(unique(model_warnings), collapse = " | ")
  )

  emmeans_rows[[outcome]] <- emm_full
  pairwise_time_rows[[outcome]] <- pair_time
  pairwise_group_rows[[outcome]] <- pair_group
  inclusion_rows[[outcome]] <- tibble(
    outcome = outcome,
    outcome_label = unname(outcome_labels[[outcome]]),
    family = unname(outcome_families[[outcome]]),
    included = TRUE,
    reason = "Analysed"
  )

  summary_text <- capture.output({
    cat("Outcome:", unname(outcome_labels[[outcome]]), "\n")
    cat("Source:", data_path, "\n")
    cat("Visits:", paste(levels(dat_i$timepoint), collapse = ", "), "\n")
    cat("Transformation constant:", transform_constant, "\n\n")
    cat("Model equation:\n")
    print(formula(fit))
    cat("\nMixed-model summary:\n")
    print(summary(fit))
    cat("\nType III ANOVA with Satterthwaite degrees of freedom:\n")
    print(anova_table)
    cat("\nEstimated marginal means:\n")
    print(emm_full)
    cat("\nTime comparisons within intervention (Holm adjusted):\n")
    print(pair_time)
    cat("\nIntervention comparisons within timepoint (Holm adjusted):\n")
    print(pair_group)
    if (length(model_warnings) > 0) {
      cat("\nModel warnings:\n", paste(unique(model_warnings), collapse = "\n"), "\n")
    }
  })

  writeLines(
    summary_text,
    file.path(summary_dir, paste0("model_", safe_filename(outcome), ".txt"))
  )
}

if (length(omnibus_rows) == 0) {
  stop(
    "No mixed models were fitted. Review OUTCOME inclusion messages, column names, and missing data.",
    call. = FALSE
  )
}

omnibus_table <- bind_rows(omnibus_rows) |>
  mutate(
    timepoint_q_bh_global = p.adjust(timepoint_p, method = "BH"),
    intervention_q_bh_global = p.adjust(intervention_p, method = "BH"),
    interaction_q_bh_global = p.adjust(interaction_p, method = "BH")
  ) |>
  group_by(family) |>
  mutate(interaction_q_bh_family = p.adjust(interaction_p, method = "BH")) |>
  ungroup() |>
  arrange(interaction_q_bh_family, interaction_p)

emmeans_table <- bind_rows(emmeans_rows)
pairwise_time_table <- bind_rows(pairwise_time_rows)
pairwise_group_table <- bind_rows(pairwise_group_rows)
inclusion_table <- bind_rows(inclusion_rows)

# -----------------------------------------------------------------------------
# 7. Complete-case repeated-measures ANOVA sensitivity analysis
# -----------------------------------------------------------------------------
rm_rows <- list()

for (outcome in names(model_store)) {
  dat_i <- model_store[[outcome]]$data
  visit_names <- levels(dat_i$timepoint)

  wide_i <- dat_i |>
    select(code_number, intervention, timepoint, log_y) |>
    distinct() |>
    pivot_wider(names_from = timepoint, values_from = log_y) |>
    drop_na(all_of(visit_names))

  if (
    nrow(wide_i) < 8 ||
    n_distinct(wide_i$intervention) < 2 ||
    min(table(wide_i$intervention)) < 3
  ) next

  long_i <- wide_i |>
    pivot_longer(
      cols = all_of(visit_names),
      names_to = "timepoint",
      values_to = "log_y"
    ) |>
    mutate(
      code_number = factor(code_number),
      intervention = droplevels(factor(intervention)),
      timepoint = factor(timepoint, levels = visit_names, ordered = FALSE)
    )

  rm_fit <- tryCatch(
    afex::aov_ez(
      id = "code_number",
      dv = "log_y",
      between = "intervention",
      within = "timepoint",
      data = long_i,
      type = 3,
      anova_table = list(correction = "GG", es = "pes")
    ),
    error = function(e) e
  )

  if (inherits(rm_fit, "error")) {
    rm_rows[[outcome]] <- tibble(
      outcome = outcome,
      outcome_label = unname(outcome_labels[[outcome]]),
      family = unname(outcome_families[[outcome]]),
      visits = paste(visit_names, collapse = ", "),
      n_complete = nrow(wide_i),
      error = conditionMessage(rm_fit)
    )
    next
  }

  rm_table <- as.data.frame(rm_fit$anova_table)
  rm_interaction_row <- first_matching_row(
    rm_table,
    c("intervention:timepoint", "timepoint:intervention")
  )

  if (is.na(rm_interaction_row)) next

  f_col <- first_matching_column(rm_table, "^F$")
  p_col <- first_matching_column(rm_table, "Pr")
  df1_col <- first_matching_column(rm_table, "num.*Df|num.*df")
  df2_col <- first_matching_column(rm_table, "den.*Df|den.*df")
  pes_col <- first_matching_column(rm_table, "pes")

  rm_rows[[outcome]] <- tibble(
    outcome = outcome,
    outcome_label = unname(outcome_labels[[outcome]]),
    family = unname(outcome_families[[outcome]]),
    visits = paste(visit_names, collapse = ", "),
    n_complete = nrow(wide_i),
    n_placebo = sum(wide_i$intervention == "Placebo"),
    n_synbiotic = sum(wide_i$intervention == "Synbiotic"),
    interaction_F = if (!is.na(f_col)) as.numeric(rm_table[rm_interaction_row, f_col]) else NA_real_,
    interaction_df1_GG = if (!is.na(df1_col)) as.numeric(rm_table[rm_interaction_row, df1_col]) else NA_real_,
    interaction_df2_GG = if (!is.na(df2_col)) as.numeric(rm_table[rm_interaction_row, df2_col]) else NA_real_,
    interaction_p_GG = if (!is.na(p_col)) as.numeric(rm_table[rm_interaction_row, p_col]) else NA_real_,
    interaction_partial_eta2 = if (!is.na(pes_col)) as.numeric(rm_table[rm_interaction_row, pes_col]) else NA_real_,
    error = NA_character_
  )
}

rm_anova_table <- bind_rows(rm_rows)

if (nrow(rm_anova_table) > 0 && "interaction_p_GG" %in% names(rm_anova_table)) {
  rm_anova_table <- rm_anova_table |>
    mutate(interaction_q_bh_global = p.adjust(interaction_p_GG, method = "BH")) |>
    group_by(family) |>
    mutate(interaction_q_bh_family = p.adjust(interaction_p_GG, method = "BH")) |>
    ungroup() |>
    arrange(interaction_q_bh_family, interaction_p_GG)
}

# -----------------------------------------------------------------------------
# 8. High-resolution outcome figures
# -----------------------------------------------------------------------------
# Okabe-Ito colour-blind-safe palette.
group_colours <- c("Placebo" = "#D55E00", "Synbiotic" = "#0072B2")
group_shapes <- c("Placebo" = 16, "Synbiotic" = 17)
group_lines <- c("Placebo" = "dashed", "Synbiotic" = "solid")

figure_theme <- theme_classic(base_size = 14, base_family = "Arial") +
  theme(
    plot.title = element_text(face = "bold", size = 16, color = "#172B4D"),
    plot.subtitle = element_text(size = 11, color = "#44546A", margin = margin(b = 8)),
    plot.caption = element_text(size = 9, color = "#5B6573", hjust = 0),
    axis.title = element_text(face = "bold", color = "#1F2937"),
    axis.text = element_text(color = "#1F2937"),
    axis.text.x = element_text(angle = 0, vjust = 0.5),
    legend.position = "top",
    legend.title = element_blank(),
    legend.justification = "left",
    panel.grid.major.y = element_line(color = "#E7ECF0", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    plot.margin = margin(12, 16, 12, 12)
  )

plot_store <- list()

for (outcome in names(model_store)) {
  visit_levels <- levels(model_store[[outcome]]$data$timepoint)
  plot_data <- model_store[[outcome]]$emmeans |>
    mutate(
      timepoint = factor(timepoint, levels = visit_levels, ordered = FALSE),
      intervention = factor(intervention, levels = c("Placebo", "Synbiotic"))
    )

  result_row <- omnibus_table |>
    filter(.data$outcome == .env$outcome) |>
    slice(1)

  subtitle_text <- paste0(
    "Time × intervention: p = ", format_p(result_row$interaction_p),
    "; BH-FDR q = ", format_p(result_row$interaction_q_bh_family)
  )

  p <- ggplot(
    plot_data,
    aes(
      x = timepoint,
      y = emmean_backtransformed,
      group = intervention,
      colour = intervention,
      shape = intervention,
      linetype = intervention
    )
  ) +
    geom_line(linewidth = 1.15) +
    geom_errorbar(
      aes(
        ymin = lower_95_backtransformed,
        ymax = upper_95_backtransformed
      ),
      width = 0.12,
      linewidth = 0.8
    ) +
    geom_point(size = 3.5, stroke = 0.9, fill = "white") +
    scale_colour_manual(values = group_colours, drop = FALSE) +
    scale_shape_manual(values = group_shapes, drop = FALSE) +
    scale_linetype_manual(values = group_lines, drop = FALSE) +
    scale_x_discrete(drop = TRUE) +
    scale_y_continuous(labels = label_number(accuracy = 0.01), expand = expansion(mult = c(0.08, 0.15))) +
    labs(
      title = unname(outcome_labels[[outcome]]),
      subtitle = subtitle_text,
      x = "Study visit",
      y = unname(outcome_labels[[outcome]]),
      caption = "Back-transformed estimated marginal means with 95% confidence intervals."
    ) +
    figure_theme

  plot_store[[outcome]] <- p
  figure_name <- paste0(sprintf("%02d", match(outcome, available_outcomes)), "_", safe_filename(outcome))

  ggsave(
    filename = file.path(figure_png_dir, paste0(figure_name, ".png")),
    plot = p,
    width = 7.5,
    height = 5.5,
    units = "in",
    dpi = 600,
    bg = "white"
  )

  ggsave(
    filename = file.path(figure_tiff_dir, paste0(figure_name, ".tiff")),
    plot = p,
    width = 7.5,
    height = 5.5,
    units = "in",
    dpi = 600,
    compression = "lzw",
    bg = "white"
  )
}

if (length(plot_store) > 0) {
  grDevices::pdf(
    file.path(out_dir, "ALL_outcome_figures_vector.pdf"),
    width = 7.5,
    height = 5.5,
    onefile = TRUE,
    useDingbats = FALSE
  )
  for (outcome in names(plot_store)) print(plot_store[[outcome]])
  grDevices::dev.off()
}

# -----------------------------------------------------------------------------
# 9. Save tables and reproducibility information
# -----------------------------------------------------------------------------
write.csv(omnibus_table, file.path(csv_dir, "MASTER_mixed_model_omnibus.csv"), row.names = FALSE)
write.csv(emmeans_table, file.path(csv_dir, "MASTER_estimated_marginal_means.csv"), row.names = FALSE)
write.csv(pairwise_time_table, file.path(csv_dir, "MASTER_pairwise_time_within_group.csv"), row.names = FALSE)
write.csv(pairwise_group_table, file.path(csv_dir, "MASTER_pairwise_group_within_timepoint.csv"), row.names = FALSE)
write.csv(rm_anova_table, file.path(csv_dir, "MASTER_repeated_measures_ANOVA.csv"), row.names = FALSE)
write.csv(inclusion_table, file.path(csv_dir, "OUTCOME_inclusion_and_exclusions.csv"), row.names = FALSE)
write.csv(cell_counts <- data |> count(timepoint, intervention, name = "n_rows"),
          file.path(csv_dir, "SAMPLE_COUNTS_by_visit_and_group.csv"), row.names = FALSE)

writeLines(
  c(
    paste("Source file:", data_path),
    paste("Analysis completed:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    "Primary equation: log_y ~ timepoint * intervention + (1 | code_number)",
    paste("Time coding:", paste(paste(time_codes, time_labels, sep = " = "), collapse = "; ")),
    "Intervention coding: 1 = Placebo; 2 = Synbiotic",
    paste("Minimum observations required per group at each analysed visit:", minimum_per_group_visit),
    paste("Available outcomes:", paste(available_outcomes, collapse = ", ")),
    paste("Unavailable requested outcomes:", paste(missing_outcomes, collapse = ", ")),
    "Mixed models use all available outcome observations and a participant random intercept.",
    "Repeated-measures ANOVA is an unadjusted complete-case sensitivity analysis with Greenhouse-Geisser correction.",
    "BH-FDR is reported globally and within prespecified outcome families.",
    "Pairwise comparisons use Holm adjustment within each emmeans contrast family."
  ),
  file.path(out_dir, "ANALYSIS_README.txt")
)

writeLines(capture.output(sessionInfo()), file.path(out_dir, "R_sessionInfo.txt"))

cat("\nAnalysis complete.\n")
cat("Source file: ", data_path, "\n", sep = "")
cat("Results folder: ", out_dir, "\n", sep = "")
cat("Outcomes modelled: ", nrow(omnibus_table), "\n", sep = "")
cat("Figures created: ", length(plot_store), " PNG + ", length(plot_store), " TIFF\n", sep = "")
