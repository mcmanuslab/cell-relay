#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggrepel)
  library(ragg)
  library(svglite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "Usage: NT466_4.3_Rplots.r <input.csv> <output.pdf> <output.png> <output.svg> <labels.csv>",
    call. = FALSE
  )
}

input_file <- args[[1]]
output_pdf <- args[[2]]
output_png <- args[[3]]
output_svg <- args[[4]]
labels_file <- args[[5]]

neg_fdr_cutoff <- 0.05
pos_fdr_cutoff <- 0.05
n_label_neg <- 25
n_label_pos <- 10
force_genes <- c("TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "CD58")
max_center_points <- 2000
center_xlim <- c(-0.2, 0.2)
center_ylim <- c(-0.2, 0.2)

cat("R:", R.version.string, "\n")
for (package in c("tidyverse", "ggrepel", "ragg", "svglite")) {
  cat(package, "version:", as.character(packageVersion(package)), "\n")
}

for (path in c(output_pdf, output_png, output_svg, labels_file)) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
}

df <- read_csv(input_file, show_col_types = FALSE)
df_plot <- df %>%
  filter(is.finite(B_lfc), is.finite(C_lfc)) %>%
  mutate(
    Gene_clean = str_remove(Gene, "_.*$"),
    Category = case_when(
      B_neg_fdr < neg_fdr_cutoff & C_neg_fdr < neg_fdr_cutoff ~ "neg_sig",
      B_pos_fdr < pos_fdr_cutoff & C_pos_fdr < pos_fdr_cutoff ~ "pos_sig",
      TRUE ~ "other"
    )
  )

cat("Rows retained for plotting:", nrow(df_plot), "\n")
cat("Rows removed:", nrow(df) - nrow(df_plot), "\n")

label_neg <- df_plot %>%
  filter(Category == "neg_sig") %>%
  arrange(B_lfc + C_lfc) %>%
  slice_head(n = n_label_neg)

label_pos <- df_plot %>%
  filter(Category == "pos_sig") %>%
  arrange(desc(B_lfc + C_lfc)) %>%
  slice_head(n = n_label_pos)

force_df <- df_plot %>% filter(Gene_clean %in% force_genes)
label_df <- bind_rows(label_neg, label_pos, force_df) %>%
  distinct(Gene_clean, .keep_all = TRUE)
write_csv(label_df, labels_file)

set.seed(123)
df_other <- df_plot %>% filter(Category == "other")
df_other_center <- df_other %>%
  filter(
    B_lfc >= center_xlim[1], B_lfc <= center_xlim[2],
    C_lfc >= center_ylim[1], C_lfc <= center_ylim[2]
  )
df_other_outer <- df_other %>%
  filter(
    B_lfc < center_xlim[1] | B_lfc > center_xlim[2] |
      C_lfc < center_ylim[1] | C_lfc > center_ylim[2]
  )
if (nrow(df_other_center) > max_center_points) {
  df_other_center_plot <- slice_sample(df_other_center, n = max_center_points)
} else {
  df_other_center_plot <- df_other_center
}
df_other_plot <- bind_rows(df_other_outer, df_other_center_plot)

segment_colors <- recode(
  label_df$Category,
  neg_sig = "#5d8af7",
  pos_sig = "#ed8590",
  .default = "black"
)

p <- ggplot(df_plot, aes(x = B_lfc, y = C_lfc)) +
  geom_point(
    data = df_other_plot,
    color = "grey70",
    size = 3,
    alpha = 0.8,
    stroke = 0
  ) +
  geom_point(
    data = df_plot %>% filter(Category == "neg_sig"),
    color = "#5d8af7",
    size = 3.0,
    alpha = 0.9,
    stroke = 0
  ) +
  geom_point(
    data = df_plot %>% filter(Category == "pos_sig"),
    color = "#ed8590",
    size = 3.0,
    alpha = 0.9,
    stroke = 0
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey60") +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey60") +
  geom_text_repel(
    data = label_df,
    aes(label = Gene_clean, color = Category),
    segment.color = segment_colors,
    point.padding = 0.6,
    box.padding = 0.6,
    force = 11,
    force_pull = 0.3,
    max.overlaps = Inf,
    min.segment.length = 0,
    segment.size = 0.4,
    segment.alpha = 0.8,
    xlim = c(-3.5, 2.5),
    ylim = c(-3.5, 2.5),
    show.legend = FALSE
  ) +
  scale_color_manual(
    values = c(
      neg_sig = "#5d8af7",
      pos_sig = "#ed8590",
      other = "black"
    )
  ) +
  theme_classic(base_size = 13) +
  coord_fixed(xlim = c(-3.5, 2.5), ylim = c(-3.5, 2.5)) +
  scale_x_continuous(
    breaks = seq(-3, 3, 1),
    labels = scales::number_format(accuracy = 0.1)
  ) +
  scale_y_continuous(
    breaks = seq(-3, 3, 1),
    labels = scales::number_format(accuracy = 0.1)
  ) +
  theme(
    panel.border = element_rect(
      colour = "black",
      fill = NA,
      linewidth = 0.8
    )
  )

ggsave(output_svg, plot = p, device = svglite::svglite, width = 8, height = 8)

if (Sys.info()[["sysname"]] == "Darwin") {
  quartz(file = output_pdf, type = "pdf", width = 8, height = 8)
  print(p)
  dev.off()
} else if (isTRUE(capabilities("cairo"))) {
  cairo_pdf(output_pdf, width = 8, height = 8, onefile = FALSE)
  print(p)
  dev.off()
} else {
  stop("A TrueType-capable PDF device is required for the R dotplot.", call. = FALSE)
}

agg_png(output_png, width = 8, height = 8, units = "in", res = 600, background = "white")
print(p)
dev.off()
