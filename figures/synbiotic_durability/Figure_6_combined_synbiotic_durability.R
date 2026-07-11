# ==========================================================
# COMBINED DURABILITY FIGURE
# Panel A: beta forest plots at 3 and 6 months
# Panel B: model-estimated trajectories
# Panel C: durability classification
# ==========================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(patchwork)
})

effects <- read_csv("combined_durability_summary.csv", show_col_types = FALSE)
trajectories <- read_csv("combined_durability_trajectories.csv", show_col_types = FALSE)

effects <- effects |>
  mutate(
    followup = factor(followup, levels = c("3 months", "6 months")),
    outcome = factor(outcome, levels = c("HDL-C", "LDL-C", "Insulin", "HOMA-IR"))
  )

trajectories <- trajectories |>
  mutate(
    timepoint = factor(timepoint, levels = c("Baseline", "3 months", "6 months")),
    group = factor(group, levels = c("Placebo", "Synbiotic"))
  )

pA1 <- effects |>
  filter(outcome %in% c("HOMA-IR", "Insulin")) |>
  ggplot(aes(beta, outcome, xmin = ci_low, xmax = ci_high, shape = followup)) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_errorbarh(height = 0.12, position = position_dodge(width = 0.35)) +
  geom_point(size = 2.8, position = position_dodge(width = 0.35)) +
  labs(
    title = "Insulin-resistance outcomes",
    x = expression(beta~"(log scale)"),
    y = NULL, shape = NULL
  ) +
  theme_classic(base_size = 10) +
  theme(legend.position = "bottom")

pA2 <- effects |>
  filter(outcome %in% c("LDL-C", "HDL-C")) |>
  ggplot(aes(beta, outcome, xmin = ci_low, xmax = ci_high, shape = followup)) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_errorbarh(height = 0.12, position = position_dodge(width = 0.35)) +
  geom_point(size = 2.8, position = position_dodge(width = 0.35)) +
  labs(
    title = "Lipid outcomes",
    x = expression(beta~"(mmol/L)"),
    y = NULL, shape = NULL
  ) +
  theme_classic(base_size = 10) +
  theme(legend.position = "bottom")

panel_A <- (pA1 / pA2) +
  plot_annotation(title = "A  Adjusted intervention effects")

make_trajectory <- function(outcome_name, y_label) {
  trajectories |>
    filter(outcome == outcome_name) |>
    ggplot(aes(timepoint, estimated_mean, group = group, linetype = group)) +
    annotate("rect", xmin = 2, xmax = 3, ymin = -Inf, ymax = Inf, alpha = 0.08) +
    geom_vline(xintercept = 2, linetype = "dashed", linewidth = 0.35) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2.1) +
    geom_errorbar(
      aes(ymin = lower_95_CI, ymax = upper_95_CI),
      width = 0.06, linewidth = 0.55
    ) +
    labs(title = outcome_name, x = NULL, y = y_label, linetype = NULL) +
    theme_classic(base_size = 9) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = if (outcome_name == "HOMA-IR") "bottom" else "none"
    )
}

pB1 <- make_trajectory("HOMA-IR", "HOMA-IR")
pB2 <- make_trajectory("Insulin", "Fasting insulin")
pB3 <- make_trajectory("LDL-C", "LDL-C (mmol/L)")
pB4 <- make_trajectory("HDL-C", "HDL-C (mmol/L)")

panel_B <- ((pB1 | pB2) / (pB3 | pB4)) +
  plot_annotation(title = "B  Model-estimated trajectories")

class_text <- data.frame(
  x = c(1, 2, 3),
  label = c(
    "Persistent through 6 months\n\nHOMA-IR\nβ = -0.524; q = 2.13 × 10^-7\n\nFasting insulin\nβ = -0.529; q = 3.89 × 10^-6",
    "Partially sustained\n\nLDL-C\nβ = -0.604 mmol/L\nP = 0.0167; q = 0.0500\n\nEffect remained but was attenuated.",
    "Not sustained\n\nHDL-C\nβ = 0.034 mmol/L\nP = 0.789; q = 0.881\n\nThe 3-month increase was not maintained."
  )
)

panel_C <- ggplot(class_text, aes(x, 1)) +
  geom_tile(width = 0.9, height = 0.9, fill = "white", colour = "black") +
  geom_text(aes(label = label), size = 3.1, lineheight = 0.95) +
  xlim(0.5, 3.5) + ylim(0.5, 1.5) +
  labs(title = "C  Durability classification") +
  theme_void() +
  theme(plot.title = element_text(face = "bold"))

combined <- panel_A | (panel_B / panel_C) +
  plot_layout(widths = c(0.9, 1.3))

ggsave(
  "Figure_6_combined_synbiotic_durability_R.png",
  combined, width = 15, height = 9.5, dpi = 600, bg = "white"
)

ggsave(
  "Figure_6_combined_synbiotic_durability_R.pdf",
  combined, width = 15, height = 9.5, bg = "white"
)
