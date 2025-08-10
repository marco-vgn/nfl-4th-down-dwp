#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(nfl4th)
})

IN_CSV  <- "data/interim/fourthdown_states.csv"
OUT_CSV <- "data/processed/4thdown_labeled.csv"
dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)

message("[INFO] Reading states CSV: ", IN_CSV)
states <- readr::read_csv(IN_CSV, show_col_types = FALSE) %>%
  mutate(
    season = as.integer(season),
    game_id = as.character(game_id),
    quarter = as.integer(quarter),
    game_seconds_remaining = as.numeric(game_seconds_remaining),
    half_seconds_remaining = as.numeric(half_seconds_remaining),
    yardline_100 = as.numeric(yardline_100),
    ydstogo = as.numeric(ydstogo),
    score_differential = as.numeric(score_differential),
    receive_2h_ko = as.logical(receive_2h_ko),
    timeouts_off = as.integer(timeouts_off),
    timeouts_def = as.integer(timeouts_def),
    roof = as.character(roof),
    surface = as.character(surface),
    temp_f = as.numeric(temp_f),
    wind_mph = as.numeric(wind_mph),
    spread_line = as.numeric(spread_line),
    down = as.integer(down),
    season_type = as.character(season_type),
    posteam = as.character(posteam),
    defteam = as.character(defteam),
    home_team = as.character(home_team),
    away_team = as.character(away_team),
    week = suppressWarnings(as.integer(week))
  )

# Derive quarter_seconds_remaining (helps some nfl4th versions)
states <- states %>%
  mutate(
    quarter_seconds_remaining = case_when(
      quarter %in% 1:4 ~ pmax(0, game_seconds_remaining - (4L - quarter) * 900L),
      quarter >= 5     ~ pmin(600, pmax(0, game_seconds_remaining)),
      TRUE             ~ NA_real_
    )
  )

# Derive home_opening_kickoff from receive_2h_ko + teams
sec_recv <- states %>%
  filter(quarter <= 2, receive_2h_ko == TRUE) %>%
  distinct(game_id, second_half_receiver = posteam)

states <- states %>%
  left_join(sec_recv, by = "game_id") %>%
  mutate(
    home_opening_kickoff = second_half_receiver == away_team
  )

# Add type (reg/post) if some helpers use it
states <- states %>%
  mutate(type = if_else(tolower(season_type) == "reg", "reg", "post"))

# Defensive filter: drop rows missing critical join/time keys
states <- states %>%
  filter(
    !is.na(home_team), !is.na(away_team), !is.na(week),
    !is.na(quarter_seconds_remaining),
    !is.na(home_opening_kickoff)
  )

message("[INFO] Running nfl4th::add_4th_probs()...")
df <- states %>%
  rename(
    qtr = quarter,
    posteam_timeouts_remaining = timeouts_off,
    defteam_timeouts_remaining = timeouts_def,
    temp = temp_f,
    wind = wind_mph
  ) %>%
  nfl4th::add_4th_probs()

message("[INFO] Label columns present:")
print(names(df))

# Map returned columns to expected names (handles version differences)
if (!"wp_go" %in% names(df) && "go_wp" %in% names(df))     df <- df %>% mutate(wp_go   = go_wp)
if (!"wp_fg" %in% names(df) && "fg_wp" %in% names(df))     df <- df %>% mutate(wp_fg   = fg_wp)
if (!"wp_punt" %in% names(df) && "punt_wp" %in% names(df)) df <- df %>% mutate(wp_punt = punt_wp)

if (!"p_fg_make" %in% names(df) && "fg_make_prob" %in% names(df)) df <- df %>% mutate(p_fg_make = fg_make_prob)
if (!"p_go_convert" %in% names(df) && "first_down_prob" %in% names(df)) df <- df %>% mutate(p_go_convert = first_down_prob)

# Fallback for current WP if nfl4th doesn't return it explicitly
if (!"wp" %in% names(df)) {
  df <- df %>% mutate(wp = pmax(wp_go, wp_fg, wp_punt, na.rm = TRUE))
}

# punt_net_yds not always provided; set NA if absent
if (!"punt_net_yds" %in% names(df)) {
  df <- df %>% mutate(punt_net_yds = NA_real_)
}

stopifnot(all(c("wp_go","wp_fg","wp_punt","wp") %in% names(df)))

out <- df %>%
  mutate(
    dwp_go   = wp_go   - wp,
    dwp_fg   = wp_fg   - wp,
    dwp_punt = wp_punt - wp
  ) %>%
  rename(
    quarter = qtr,
    timeouts_off = posteam_timeouts_remaining,
    timeouts_def = defteam_timeouts_remaining,
    temp_f = temp,
    wind_mph = wind
  ) %>%
  select(
    season, game_id, play_id,
    quarter, game_seconds_remaining, half_seconds_remaining,
    yardline_100, ydstogo, score_differential,
    receive_2h_ko, timeouts_off, timeouts_def,
    roof, surface, temp_f, wind_mph, spread_line,
    wp, wp_go, wp_fg, wp_punt,
    dwp_go, dwp_fg, dwp_punt,
    p_fg_make, p_go_convert, punt_net_yds
  )

message("[INFO] Writing labels CSV: ", OUT_CSV)
readr::write_csv(out, OUT_CSV)
message("[DONE] Rows: ", nrow(out))
