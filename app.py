from mesa.visualization import SolaraViz, make_plot_component
import solara
from matplotlib.figure import Figure
import seaborn as sns

from mesa.visualization.utils import update_counter

from model import *

model = Economy()

model_params = {
    "H": {
        "type": "SliderInt",
        "value": 1000,
        "label": "Number of agents:",
        "min": 10,
        "max": 3000,
        "step": 1,
    },
}

# `employment` is named in the DataCollector method.
# The steps in this plot refer to months (so 21 actuall steps each "step").
EmploymentPlot = make_plot_component("employment")

# Create a custom chart.
def FirmSizeDensity(model):
    update_counter.get()  # This is required to update the counter
    # Note: you must initialize a figure using this method instead of
    # plt.figure(), for thread safety purpose
    fig = Figure()
    ax = fig.subplots()
    # Gather the data.
    data = [len(firm.employees) for firm in model.agents_by_type[Firm]]
    # Seaborn directly modifies the ax attribute. Specify ax argument for thread safety.
    sns.kdeplot(data, ax=ax)
    # Note: you have to use Matplotlib's OOP API instead of plt.hist
    # because plt.hist is not thread-safe.
    solara.FigureMatplotlib(fig)

page = SolaraViz(
    model,
    components=[EmploymentPlot, FirmSizeDensity],
    model_params=model_params,
    name="Macroeconomic Agent-Based Model",
)