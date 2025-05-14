library(tidyverse)

# Set the years to display for time series
time_series_years = 50
# Years to drop in sample
drop_years = 100

# Import household data
hdata = read.csv("../output/hdata.csv")%>%
  pivot_wider(
    id_cols = c("Step", "AgentID"),
    names_from = variable,
    values_from = value
  )

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

unemp_plot



fdata = read.csv("../output/fdata.csv")%>%
  pivot_wider(
    id_cols = c("Step", "AgentID"),
    names_from = variable,
    values_from = value
  )
