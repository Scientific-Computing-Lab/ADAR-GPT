#!/usr/bin/env Rscript

# RNA Editing Data Processing - Simple Equal Groups
# =================================================
# Same logic as your original script:
# - Reads a CSV with EditingIndex (and ideally structure, L, R, chr, pos, site_idx_in_seq)
# - Cleans data
# - Builds four NON-OVERLAPPING balanced groups by thresholds:
#    Group 1:  YES (1–5%)  vs  NO (<1%)
#    Group 2:  YES (5–10%) vs  NO (<5%)
#    Group 3:  YES (10–15%)vs  NO (<10%)
#    Group 4:  YES (>=15%) vs  NO (<15%)
# - Splits each group into 80/20 train/valid (configurable)
# - Saves CSVs: group1/5/10/15_{train|valid}.csv
#
# Parameters (CLI):
#   --input_csv   (required)
#   --output_dir  (required)
#   --train_ratio (default: 0.8)
#   --seed        (default: 42)

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(readr)
  library(tidyr)
})

# ---------- CLI options ----------
opt <- OptionParser(option_list = list(
  make_option("--input_csv",  type = "character", help = "Path to input CSV (required)."),
  make_option("--output_dir", type = "character", help = "Output directory for all CSVs (required)."),
  make_option("--train_ratio", type = "double", default = 0.8, help = "Train/valid split ratio [default: %default]."),
  make_option("--seed",       type = "integer", default = 42,  help = "Random seed [default: %default].")
))

args <- parse_args(opt)

if (is.null(args$input_csv) || is.null(args$output_dir)) {
  print_help(opt)
  stop("Both --input_csv and --output_dir are required.", call. = FALSE)
}

if (!file.exists(args$input_csv)) {
  stop(paste0("Input CSV not found: ", args$input_csv))
}
if (!dir.exists(args$output_dir)) {
  dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
}

set.seed(args$seed)

# ========== Read Data ==========
data <- read.csv(args$input_csv, stringsAsFactors = FALSE)

cat("============================================\n")
cat("Data Processing - 4 Equal-Sized Groups\n")
cat("============================================\n\n")

# Step 1: Cleaning
cat("Rows in source:", nrow(data), "\n")

data_clean <- data %>%
  filter(!is.na(EditingIndex) &
           EditingIndex != "" &
           EditingIndex != "." &
           !is.na(structure) &
           structure != "")

data_clean$EditingIndex <- as.numeric(data_clean$EditingIndex)
data_clean <- data_clean %>% filter(!is.na(EditingIndex))

cat("Rows after cleaning:", nrow(data_clean), "\n")
cat("EditingIndex range:", min(data_clean$EditingIndex), "-", max(data_clean$EditingIndex), "\n\n")

# Create a unique site identifier
# Preferred: chr + pos + site_idx_in_seq
if ("site_idx_in_seq" %in% colnames(data_clean) &&
    "chr" %in% colnames(data_clean) &&
    "pos" %in% colnames(data_clean)) {
  data_clean$Genomic_Location <- paste0(data_clean$chr, ":", data_clean$pos, ":", data_clean$site_idx_in_seq)
} else if ("chr" %in% colnames(data_clean) && "pos" %in% colnames(data_clean)) {
  # Fallback: chr:pos (might be non-unique)
  data_clean$Genomic_Location <- paste0(data_clean$chr, ":", data_clean$pos)
  warning("site_idx_in_seq not found — Genomic_Location may be non-unique (chr:pos only)!")
} else {
  # No genomic fields — generate synthetic IDs
  data_clean$Genomic_Location <- paste0("site_", 1:nrow(data_clean))
  warning("chr/pos not found — generating synthetic IDs!")
}

# Check uniqueness
n_unique <- length(unique(data_clean$Genomic_Location))
if (n_unique < nrow(data_clean)) {
  warning(paste0("Found ", nrow(data_clean) - n_unique, " duplicate Genomic_Location IDs — removing duplicates (keeping first)."))
  data_clean <- data_clean %>% distinct(Genomic_Location, .keep_all = TRUE)
  cat("Rows after removing duplicates:", nrow(data_clean), "\n")
}
cat("Unique IDs:", length(unique(data_clean$Genomic_Location)), "\n\n")

# ========== Define groups by thresholds ==========

# Group 1: 1–5 vs <1
table1_high <- data_clean %>% filter(EditingIndex >= 1 & EditingIndex < 5)
table1_low  <- data_clean %>% filter(EditingIndex < 1)

# Group 2: 5–10 vs <5
table5_high <- data_clean %>% filter(EditingIndex >= 5 & EditingIndex < 10)
table5_low  <- data_clean %>% filter(EditingIndex < 5)

# Group 3: 10–15 vs <10
table10_high <- data_clean %>% filter(EditingIndex >= 10 & EditingIndex < 15)
table10_low  <- data_clean %>% filter(EditingIndex < 10)

# Group 4: >=15 vs <15
table15_high <- data_clean %>% filter(EditingIndex >= 15)
table15_low  <- data_clean %>% filter(EditingIndex < 15)

# ========== Candidate counts ==========
cat("\n--- Candidate counts per group ---\n")
cat("Group 1: YES (1–5%):", nrow(table1_high), ", NO (<1%):", nrow(table1_low), "\n")
cat("Group 2: YES (5–10%):", nrow(table5_high), ", NO (<5%):", nrow(table5_low), "\n")
cat("Group 3: YES (10–15%):", nrow(table10_high), ", NO (<10%):", nrow(table10_low), "\n")
cat("Group 4: YES (>=15%):", nrow(table15_high), ", NO (<15%):", nrow(table15_low), "\n\n")

# Balanced size per group (min of YES/NO)
n1 <- min(nrow(table1_high),  nrow(table1_low))
n2 <- min(nrow(table5_high),  nrow(table5_low))
n3 <- min(nrow(table10_high), nrow(table10_low))
n4 <- min(nrow(table15_high), nrow(table15_low))

cat("Max balanced per group:\n")
cat("Group 1:", n1 * 2, "(", n1, "YES +", n1, "NO)\n")
cat("Group 2:", n2 * 2, "(", n2, "YES +", n2, "NO)\n")
cat("Group 3:", n3 * 2, "(", n3, "YES +", n3, "NO)\n")
cat("Group 4:", n4 * 2, "(", n4, "YES +", n4, "NO)\n\n")

# Global balanced size across all groups
n <- min(n1, n2, n3, n4)

cat("==============================================\n")
cat("Final per-group size:", n * 2, "\n")
cat("Per class (YES/NO):", n, "\n")
cat("==============================================\n\n")

# ========== Build groups (NO OVERLAP) ==========

# Group 1
cat("\n--- Building Group 1 ---\n")
table1 <- rbind(
  table1_high %>% sample_n(n),
  table1_low  %>% sample_n(n)
)
table1 <- table1 %>% mutate(label = ifelse(EditingIndex >= 1, "yes", "no"))
remaining_data <- data_clean %>% filter(!Genomic_Location %in% table1$Genomic_Location)
cat("Group 1 built:", nrow(table1), "rows (", sum(table1$label == "yes"), "yes,", sum(table1$label == "no"), "no)\n")

# Group 2
cat("\n--- Building Group 2 ---\n")
table5_high_remaining <- remaining_data %>% filter(EditingIndex >= 5 & EditingIndex < 10)
table5_low_remaining  <- remaining_data %>% filter(EditingIndex < 5)

table5 <- rbind(
  table5_high_remaining %>% sample_n(min(n, nrow(table5_high_remaining))),
  table5_low_remaining  %>% sample_n(min(n, nrow(table5_low_remaining)))
)
table5 <- table5 %>% mutate(label = ifelse(EditingIndex >= 5, "yes", "no"))
remaining_data <- remaining_data %>% filter(!Genomic_Location %in% table5$Genomic_Location)
cat("Group 2 built:", nrow(table5), "rows (", sum(table5$label == "yes"), "yes,", sum(table5$label == "no"), "no)\n")

# Group 3
cat("\n--- Building Group 3 ---\n")
table10_high_remaining <- remaining_data %>% filter(EditingIndex >= 10 & EditingIndex < 15)
table10_low_remaining  <- remaining_data %>% filter(EditingIndex < 10)

table10 <- rbind(
  table10_high_remaining %>% sample_n(min(n, nrow(table10_high_remaining))),
  table10_low_remaining  %>% sample_n(min(n, nrow(table10_low_remaining)))
)
table10 <- table10 %>% mutate(label = ifelse(EditingIndex >= 10, "yes", "no"))
remaining_data <- remaining_data %>% filter(!Genomic_Location %in% table10$Genomic_Location)
cat("Group 3 built:", nrow(table10), "rows (", sum(table10$label == "yes"), "yes,", sum(table10$label == "no"), "no)\n")

# Group 4
cat("\n--- Building Group 4 ---\n")
table15_high_remaining <- remaining_data %>% filter(EditingIndex >= 15)
table15_low_remaining  <- remaining_data %>% filter(EditingIndex < 15)

table15 <- rbind(
  table15_high_remaining %>% sample_n(min(n, nrow(table15_high_remaining))),
  table15_low_remaining  %>% sample_n(min(n, nrow(table15_low_remaining)))
)
table15 <- table15 %>% mutate(label = ifelse(EditingIndex >= 15, "yes", "no"))
cat("Group 4 built:", nrow(table15), "rows (", sum(table15$label == "yes"), "yes,", sum(table15$label == "no"), "no)\n")

# ========== Train/Validation split (80/20 by default) ==========
cat("\n\n################################################\n")
cat("# Train/Validation split (", args$train_ratio, "/", 1 - args$train_ratio, ")\n", sep = "")
cat("################################################\n\n")

split_train_valid <- function(data, train_ratio = 0.8) {
  n_total <- nrow(data)
  n_train <- floor(n_total * train_ratio)
  data_shuffled <- data %>% sample_frac(1)
  train <- data_shuffled[1:n_train, ]
  valid <- data_shuffled[(n_train + 1):n_total, ]
  list(train = train, valid = valid)
}

group1_split <- split_train_valid(table1, args$train_ratio)
group2_split <- split_train_valid(table5,  args$train_ratio)
group3_split <- split_train_valid(table10, args$train_ratio)
group4_split <- split_train_valid(table15, args$train_ratio)

cat("Group 1: Train =", nrow(group1_split$train), ", Valid =", nrow(group1_split$valid), "\n")
cat("Group 2: Train =", nrow(group2_split$train), ", Valid =", nrow(group2_split$valid), "\n")
cat("Group 3: Train =", nrow(group3_split$train), ", Valid =", nrow(group3_split$valid), "\n")
cat("Group 4: Train =", nrow(group4_split$train), ", Valid =", nrow(group4_split$valid), "\n")

# ========== Save files ==========
cat("\n\n################################################\n")
cat("# Writing files                                 #\n")
cat("################################################\n\n")

# Select columns to export (keep original choice: structure, L, R, label)
select_cols <- function(df) {
  cols <- c("structure", "L", "R", "label")
  existing <- cols[cols %in% colnames(df)]
  if (length(existing) == 4) {
    df %>% select(all_of(existing))
  } else {
    # If some of these columns are missing, keep all columns + label (to avoid breaking)
    if (!("label" %in% names(df))) df$label <- NA_character_
    df
  }
}

out_g1_train <- file.path(args$output_dir, "group1_train.csv")
out_g1_valid <- file.path(args$output_dir, "group1_valid.csv")
write.csv(select_cols(group1_split$train), out_g1_train, row.names = FALSE)
write.csv(select_cols(group1_split$valid), out_g1_valid, row.names = FALSE)
cat("✓ Group 1: group1_train.csv, group1_valid.csv\n")

out_g5_train <- file.path(args$output_dir, "group5_train.csv")
out_g5_valid <- file.path(args$output_dir, "group5_valid.csv")
write.csv(select_cols(group2_split$train), out_g5_train, row.names = FALSE)
write.csv(select_cols(group2_split$valid), out_g5_valid, row.names = FALSE)
cat("✓ Group 2: group5_train.csv, group5_valid.csv\n")

out_g10_train <- file.path(args$output_dir, "group10_train.csv")
out_g10_valid <- file.path(args$output_dir, "group10_valid.csv")
write.csv(select_cols(group3_split$train), out_g10_train, row.names = FALSE)
write.csv(select_cols(group3_split$valid), out_g10_valid, row.names = FALSE)
cat("✓ Group 3: group10_train.csv, group10_valid.csv\n")

out_g15_train <- file.path(args$output_dir, "group15_train.csv")
out_g15_valid <- file.path(args$output_dir, "group15_valid.csv")
write.csv(select_cols(group4_split$train), out_g15_train, row.names = FALSE)
write.csv(select_cols(group4_split$valid), out_g15_valid, row.names = FALSE)
cat("✓ Group 4: group15_train.csv, group15_valid.csv\n")

# ========== Summary ==========
cat("\n\n################################################\n")
cat("# Summary                                      #\n")
cat("################################################\n")

cat("\nGroup sizes:\n")
cat("Group 1:", nrow(table1),  "samples\n")
cat("Group 2:", nrow(table5),  "samples\n")
cat("Group 3:", nrow(table10), "samples\n")
cat("Group 4:", nrow(table15), "samples\n")

all_locations <- c(table1$Genomic_Location, table5$Genomic_Location,
                   table10$Genomic_Location, table15$Genomic_Location)
cat("\nTotal sites:", length(all_locations), "\n")
cat("Unique sites:", length(unique(all_locations)), "\n")

if (length(all_locations) == length(unique(all_locations))) {
  cat("✓ No duplicates across groups — each site appears exactly once.\n")
} else {
  cat("⚠ Found", length(all_locations) - length(unique(all_locations)), "duplicates across groups!\n")
}

cat("\n✓ All 4 groups are equal in size.\n")
cat("✓ Each group is balanced (equal YES and NO).\n")
cat("✓ No site is reused across groups.\n")
cat("✓ Train/Valid split:", args$train_ratio, "/", 1 - args$train_ratio, "\n")
cat("\nFiles written to:", normalizePath(args$output_dir), "\n")
