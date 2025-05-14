library(tidyverse)

# Set the years to display for time series.
time_series_years = 50
# Years to drop in sample.
drop_years = 100

# Import household data.
hdata = read.csv("../../output/hdata.csv")%>%
  pivot_wider(
    id_cols = c("Step", "AgentID"),
    names_from = variable,
    values_from = value
  )

# Recreate Fig 4b.
unemp_data = hdata %>%
  group_by(Step) %>%
  summarize(
    employment = sum(employment)
  )  %>%
  # Years * Months
  slice_tail(n = 12*time_series_years) %>%
  mutate(
    month = row_number(),
    year = month/12
    )

unemp_plot = unemp_data %>%
  ggplot(aes(x = year, y = employment)) +
  geom_line() +
  labs(
    y = "Employed Households",
    x = "Years"
  )
  

# Import firm data.
fdata = read.csv("../../output/fdata.csv")%>%
  pivot_wider(
    id_cols = c("Step", "AgentID"),
    names_from = variable,
    values_from = value
  )

# Recreate Fig 4a.
hdata_series = hdata %>%
  group_by(Step) %>%
  summarize(
    employment = sum(employment),
    full_demand = sum(full_demand)
  )

fdata_series = fdata %>%
  group_by(Step) %>%
  summarize(
    output = sum(output),
    price = mean(price),
    employed = sum(employees),
    fulfilled_demand = sum(fulfilled_demand),
    vacancy = sum(vacancy),
    inventory = sum(inventory),
    wage = mean(wage)
  )

full_series = fdata_series %>%
  select(-Step) %>%
  cbind(hdata_series, .) %>%
  filter(row_number() > drop_years*12) %>%
  mutate(
    month = row_number(),
    year = month/12,
    employ_check = employment - employed,
    price_chg_month = price - lag(price),
    price_chg_year = price - lag(price, 12),
    unemp = 1000 - employment
  )
# sum(full_series$employ_check)

unfulfilled_plot = full_series %>%
  mutate(
    unfulfilled = 1 - fulfilled_demand/full_demand
  )%>%
  ggplot(aes(x = unfulfilled*100)) +
  geom_density() +
  labs(
    y = "Probability density function",
    x = "Unsatisfied demand (in %)"
  )# +
  # scale_x_continuous(limits = c(0, 0.2))

# Recreate Fig 7a.
ccf = forecast::ggCcf(
  full_series$price, full_series$output, lag.max = 48, plot = TRUE
  )

ccf_plot = ccf +
  labs(
    title = "",
    y = "Cross-correlation between output and lags of mean price",
    x = "Lags of mean price (in months)"
  ) +
  annotate(
    "rect", 
    xmin = -24, xmax = 24, 
    ymin = -Inf, ymax = Inf, 
    alpha = 0.2, fill = "blue"
  )
  

full_series_noisy = full_series %>%
  mutate(
    unemp = unemp + runif(1000, -0.5, 0.5),
    vacancy = vacancy + runif(1000, -0.5, 0.5)
  )

# Recreate Fig 5a.
phillips_plot = full_series_noisy %>% ggplot(aes(x = price_chg_year, y = unemp)) +
  geom_point() +
  labs(
    y = "Unemployed households",
    x = "First-difference of mean price across firms"
  )


# Recreate Fig 5b.
beveridge_plot = full_series_noisy %>% ggplot(aes(x = vacancy, y = unemp)) +
  geom_point() +
  labs(
    y = "Unemployed households",
    x = "Total vacancies"
  )

# Recreate Fig 6a.
fslice_data = fdata %>%
  group_by(AgentID) %>%
  arrange(AgentID, Step) %>%
  slice_tail(n = 13) %>%
  mutate(
    price_chg_month = price - lag(price),
    price_changed = ifelse(price_chg_month == 0, 0, 1)
    ) %>%
  slice_tail(n = 1)

size_plot = fslice_data %>%
  ggplot(aes(x = fulfilled_demand)) +
  geom_density() +
  labs(
    y = "Probability density function",
    x = "Firm's fulfilled demand as of the last recorded step"
  )

size_plot

# Recreate Fig 6b.
price_changed_data = fdata %>%
  filter(row_number() > drop_years*12) %>%
  group_by(AgentID) %>%
  arrange(AgentID, Step) %>%
  mutate(
    price_chg_month = price - lag(price),
    price_chg_month_rel = 100*(price/lag(price) - 1),
    price_changed = ifelse(price_chg_month == 0, 0, 1)
  )  %>%
  ungroup() 

firms_changing_price_plot_1 = price_changed_data %>% 
  ggplot(aes(x = price_chg_month_rel)) +
  geom_density()

firms_changing_price_plot_2 = price_changed_data %>%
  group_by(AgentID) %>%
  slice_tail(n = 1) %>% 
  ggplot(aes(x = price_chg_month_rel)) +
  geom_density()

firms_changing_price_plot_3 = price_changed_data %>%
  group_by(Step) %>%
  summarize(
    percent_who_changed_price = sum(price_changed)
  ) %>%
  ggplot(aes(x = percent_who_changed_price)) +
  geom_density()
  
