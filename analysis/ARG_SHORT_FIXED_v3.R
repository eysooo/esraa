# ARG longitudinal interaction regression: Baseline, 3, 6, 9 and 12 months
# Run the complete file with RStudio's Source button.
message("RUNNING ARG SHORT FIXED VERSION 3")

packages <- c("readxl", "janitor", "dplyr", "tidyr", "lmerTest", "emmeans", "afex", "ggplot2")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Install first: install.packages(c(", paste(sprintf('"%s"', missing), collapse = ", "), "))")

suppressPackageStartupMessages({
  library(readxl); library(janitor); library(dplyr); library(tidyr)
  library(lmerTest); library(emmeans); library(afex); library(ggplot2)
})
options(contrasts = c("contr.sum", "contr.poly"))

# 1. Read and prepare data -----------------------------------------------------
data_path <- path.expand("~/Desktop/Qatar University/PhD Project/project documents/new results /ARG FULL DATA.xlsx")
if (!file.exists(data_path)) stop("Excel file not found: ", data_path)

ARG_FULL_DATA <- read_excel(data_path, .name_repair = janitor::make_clean_names)
data <- ARG_FULL_DATA[, colSums(!is.na(ARG_FULL_DATA)) > 0, drop = FALSE]

data <- data |>
  mutate(
    code_number = factor(code_number),
    intervention = factor(intervention, levels = c(1, 2), labels = c("Placebo", "Synbiotic")),
    timepoint = factor(
      timepoint,
      levels = 0:4,
      labels = c("Baseline", "Month 3", "Month 6", "Month 9", "Month 12")
    )
  ) |>
  drop_na(code_number, intervention, timepoint)

if (anyDuplicated(data[c("code_number", "timepoint")])) {
  stop("Duplicate participant/timepoint rows found. Please correct the Excel file.")
}

y_labels <- c(
  bmi = "BMI (kg/m²)", homa_ir = "HOMA-IR",
  fasting_glucose = "Fasting glucose (mmol/L)", insulin = "Fasting insulin (µU/mL)",
  tc = "Total cholesterol (mmol/L)", tg = "Triglycerides (mmol/L)",
  hdl_c = "HDL-C (mmol/L)", ldl_c = "LDL-C (mmol/L)",
  urea = "Urea (mmol/L)", crea = "Creatinine (µmol/L)",
  t_bil_v = "Total bilirubin (µmol/L)", d_bil = "Direct bilirubin (µmol/L)",
  tp = "Total protein (g/L)", alb = "Albumin (g/L)",
  alt = "ALT (U/L)", ast = "AST (U/L)", ldh = "LDH (U/L)", g_gt = "γ-GT (U/L)"
)
outcomes <- intersect(names(y_labels), names(data))
if (!length(outcomes)) stop("No requested outcome columns were found.")

out_dir <- file.path(dirname(data_path), "ARG_longitudinal_results")
fig_dir <- file.path(out_dir, "figures_600dpi")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# 2. Mixed models: equation kept exactly unchanged ----------------------------
models <- list(); omnibus <- list(); emm_list <- list(); rm_results <- list()
get_value <- function(tab, row, choices) {
  column <- intersect(choices, names(tab))[1]
  if (!length(column)) return(NA_real_)
  as.numeric(tab[row, column])
}

for (y in outcomes) {
  d <- data |>
    transmute(
      code_number, intervention, timepoint,
      y_raw = suppressWarnings(as.numeric(gsub(",", ".", as.character(.data[[y]]), fixed = TRUE)))
    ) |>
    drop_na()

  valid_visits <- d |>
    count(timepoint, intervention) |>
    complete(timepoint, intervention, fill = list(n = 0)) |>
    group_by(timepoint) |>
    summarise(min_n = min(n), .groups = "drop") |>
    filter(min_n >= 5) |>
    pull(timepoint) |>
    as.character()

  d <- d |> filter(as.character(timepoint) %in% valid_visits) |> droplevels()
  if (nlevels(d$timepoint) < 2 || nlevels(d$intervention) < 2) next

  shift <- if (min(d$y_raw) <= 0) abs(min(d$y_raw)) + 0.1 else 0
  d$log_y <- log(d$y_raw + shift)

  # DO NOT CHANGE THIS INTERACTION-REGRESSION EQUATION.
  fit <- try(
    lmer(
      log_y ~ timepoint * intervention + (1 | code_number),
      data = d, REML = TRUE,
      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
    ),
    silent = TRUE
  )
  if (inherits(fit, "try-error")) {
    warning("Model skipped for ", y, ": ", as.character(fit))
    next
  }

  # Correct: use the anova generic; it dispatches to the lmerTest method.
  a <- anova(fit, type = 3, ddf = "Satterthwaite")
  int_row <- intersect(c("timepoint:intervention", "intervention:timepoint"), rownames(a))[1]
  p_int <- if (length(int_row)) as.numeric(a[int_row, "Pr(>F)"]) else NA_real_

  e <- as.data.frame(confint(emmeans(fit, ~ timepoint * intervention))) |>
    mutate(
      outcome = y,
      estimate = exp(emmean) - shift,
      lower_95 = exp(lower.CL) - shift,
      upper_95 = exp(upper.CL) - shift
    )

  models[[y]] <- fit
  emm_list[[y]] <- e
  omnibus[[y]] <- data.frame(
    outcome = y, n = nrow(d), participants = n_distinct(d$code_number),
    visits = paste(levels(d$timepoint), collapse = ", "),
    interaction_p = p_int, singular = isSingular(fit)
  )

  # Complete-case repeated-measures ANOVA (Greenhouse-Geisser corrected).
  visits <- levels(d$timepoint)
  wide <- d |>
    select(code_number, intervention, timepoint, log_y) |>
    pivot_wider(names_from = timepoint, values_from = log_y) |>
    drop_na(all_of(visits))

  if (nrow(wide) >= 8 && min(table(wide$intervention)) >= 3) {
    long <- wide |>
      pivot_longer(all_of(visits), names_to = "timepoint", values_to = "log_y") |>
      mutate(timepoint = factor(timepoint, levels = visits))

    rm_fit <- try(
      afex::aov_ez(
        id = "code_number", dv = "log_y", data = long,
        between = "intervention", within = "timepoint", type = 3,
        anova_table = list(correction = "GG", es = "pes")
      ),
      silent = TRUE
    )
    if (!inherits(rm_fit, "try-error")) {
      rt <- as.data.frame(rm_fit$anova_table)
      rr <- intersect(c("intervention:timepoint", "timepoint:intervention"), rownames(rt))[1]
      if (length(rr)) rm_results[[y]] <- data.frame(
        outcome = y, n_complete = nrow(wide),
        F = get_value(rt, rr, "F"),
        df1_GG = get_value(rt, rr, c("num Df", "num.Df")),
        df2_GG = get_value(rt, rr, c("den Df", "den.Df")),
        interaction_p_GG = get_value(rt, rr, c("Pr(>F)", "Pr..F.")),
        partial_eta2 = get_value(rt, rr, "pes")
      )
    }
  }
}

# 3. FDR correction and high-resolution figures -------------------------------
if (!length(omnibus)) stop("No models were fitted; check the outcome columns and missing data.")
omnibus <- bind_rows(omnibus) |>
  mutate(interaction_q_BH = p.adjust(interaction_p, "BH"))
emmeans_all <- bind_rows(emm_list)
rm_anova <- bind_rows(rm_results)
if (nrow(rm_anova)) rm_anova$interaction_q_BH <- p.adjust(rm_anova$interaction_p_GG, "BH")

colours <- c("Placebo" = "#D55E00", "Synbiotic" = "#0072B2")

for (y in names(models)) {
  e <- emm_list[[y]]
  q <- omnibus$interaction_q_BH[omnibus$outcome == y]

  p <- ggplot(e, aes(timepoint, estimate, colour = intervention, group = intervention)) +
    geom_line(linewidth = 1.2) +
    geom_errorbar(aes(ymin = lower_95, ymax = upper_95), width = 0.12, linewidth = 0.8) +
    geom_point(size = 3.5) +
    scale_colour_manual(values = colours) +
    labs(
      title = unname(y_labels[y]),
      subtitle = paste0("Time × intervention: BH-FDR q = ", signif(q, 3)),
      x = "Study visit", y = unname(y_labels[y]), colour = NULL
    ) +
    theme_classic(base_size = 15) +
    theme(
      plot.title = element_text(face = "bold", colour = "#17365D"),
      legend.position = "top", panel.grid.major.y = element_line(colour = "#E6EAF0")
    )

  ggsave(file.path(fig_dir, paste0(y, ".png")), p, width = 7.5, height = 5.5, dpi = 600, bg = "white")
  ggsave(file.path(fig_dir, paste0(y, ".tiff")), p, width = 7.5, height = 5.5, dpi = 600, compression = "lzw", bg = "white")
}

# 4. Save results --------------------------------------------------------------
write.csv(omnibus, file.path(out_dir, "mixed_model_interactions.csv"), row.names = FALSE)
write.csv(emmeans_all, file.path(out_dir, "estimated_marginal_means.csv"), row.names = FALSE)
write.csv(rm_anova, file.path(out_dir, "repeated_measures_ANOVA.csv"), row.names = FALSE)

cat("\nCompleted successfully. Results saved in:\n", out_dir, "\n")
