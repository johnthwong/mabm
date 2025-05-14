library(tidyverse)

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
  slice_tail(n = 12*50) %>%
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
