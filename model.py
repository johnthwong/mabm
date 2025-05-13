# Has multi-dimensional arrays and matrices.
# Has a large collection of mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd

# Data visualization tools.
import seaborn as sns
import mesa
import random

# Optimization tool
from scipy.optimize import minimize_scalar


def incdf(cdf):
    # Returns true if random probability falls within cdf.
    draw = random.uniform(0, 1)
    if draw <= cdf:
        return True
    else:
        return False


def employment(employer):
    if employer is None:
        return 0
    else:
        return 1


class Economy(mesa.Model):
    """
    This is the economy.

    Exogenous attributes (i.e., parameters):
        H: Number of households
        S: Number of sellers in each buyer's network.
        max_wage_chg: Upper bound of the uniform distribution from which the firm draws the wage offer change.
        slack: Months, for which all of a firm's were filled, until the firm reduces wages for all employees.
        inventory_low: Multiplier on `demand` to determine minimum desirable inventory level.
        inventory_high: Multiplier on `demand` to determine maximum desirable inventory level.
        price_low: Multiplier on `wage` to determine minimum desirable price.
        price_high: Multipler on `wage` to determine maximum desirable price.
        tech_param: Parameter for production function.
        buffer: Multiplier on total wage bill to determine amount of money saved each month.
        swap_for_reliability_prob: Probability household tries to swap seller who ran out of stock for new seller.
        swap_for_price_prob: Probability household tries to swap seller for lower price.
        min_savings: Minimum change in price required for household to replace new seller for old seller.
        applys: Number of firms visited when household is unemployed.
        quit_prob: Probability that an employed but underpaid worker quits.
        aversion: Parameter for household's risk aversion for consumption.
        wage_r: Reservation wage.
        lower_wage_r: Percent by which reservation wage is lowered if a household is unemployed.
    """

    # This method intializes.
    def __init__(
        self,
        seed: int = None,
        H: int = 1000,
        S: int = 7,
        max_wage_chg=0.019,
        slack: int = 6,
        inventory_low=0.25,
        inventory_high=1,
        price_low=1.025,
        price_high=1.15,
        price_chg_prob=0.75,
        max_price_chg=0.02,
        tech_param: int = 3,
        swap_for_reliability_prob=0.25,
        swap_for_price_prob=0.25,
        min_savings: float = 0.01,
        applys=5,
        quit_prob=0.1,
        aversion=0.9,
        browse=7,
        buffer: int = 0.1,
        init_wage_r=0.1,
        lower_wage_r=0.1,
    ):
        # Initialize the parent class with relevant parameters.
        super().__init__(seed=seed)
        # Parameters.
        self.months = 0
        self.days_in_month = 21
        self.H = H
        self.F = int(H * 1e-1)  # Number of firms
        self.S = S
        self.max_wage_chg = max_wage_chg
        self.slack = slack
        self.inventory_low = inventory_low
        self.inventory_high = inventory_high
        self.price_low = price_low
        self.price_high = price_high
        self.price_chg_prob = price_chg_prob
        self.max_price_chg = max_price_chg
        self.tech_param = tech_param
        self.swap_for_reliability_prob = swap_for_reliability_prob
        self.swap_for_price_prob = swap_for_price_prob
        self.min_savings = min_savings
        self.applys = applys
        self.quit_prob = quit_prob
        self.aversion = aversion
        self.browse = browse
        self.buffer = buffer
        self.init_wage_r = init_wage_r
        self.lower_wage_r = lower_wage_r
        
        # Create households.
        Household.create_agents(model=self, n=self.H)

        # Households create singleton firms.
        self.agents_by_type[Household].do("initialize_singletons")

        # Optimize effort
        self.agents_by_type[Household].do("optimize_current")

        # Connect labor market
        self.agents_by_type[Household].shuffle_do("connect_labor_market")

        # Make moves
        self.agents_by_type[Household].shuffle_do("make_moves")
        
        # Pair households to firms.
        self.agents_by_type[Household].shuffle_do("assign_firms", S=self.S)

        # Initialize reservation wage.
        self.agents_by_type[Household].do(
            "initialize_reservation",
            init_wage_r=self.init_wage_r,
            tech_param=self.tech_param,
            days_in_month=self.days_in_month,
        )

        # Initialize money.
        self.agents_by_type[Household].do(
            "initialize_money",
        )

        # Initialize price and wages.
        self.agents_by_type[Firm].do(
            "initialize_price_wage",
            tech_param=self.tech_param,
            days_in_month=self.days_in_month,
        )

        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            agent_reporters={"money": "money"},
            agenttype_reporters={
                Household: {
                    "employment": lambda a: employment(a.employer),
                },
                Firm: {
                    "output": lambda a: a.month_output,
                    "price": lambda a: a.price,
                    "employees": lambda a: len(a.employees),
                    "demand": lambda a: a.fulfilled_demand,
                    "vacancy": lambda a: a.opening_hist[-1],
                    "inventory": lambda a: a.inventory,
                    "wage": lambda a: a.wage,
                },
            },
        )

    # This method checks if day is end of month.
    def end(self):
        if self.steps % self.days_in_month == 0:
            return True

    # This method checks if day is start of month.
    def start(self):
        if self.steps % self.days_in_month == 1:
            return True

    # This method records the month.
    def record_month(self):
        self.months += 1

    # This method represents all the action that takes place in one step of the model.
        
    # Start of month activities.
    def step(self):

        print(f"Today is day {self.steps}.")
        if self.start():
            print("start of month.")

            self.record_month()

            # Households recall whether they were employed last month.
            self.agents_by_type[Household].do("update_employment_hist")
            # Firms set wages.
            self.agents_by_type[Firm].do(
                "set_wage",
                max_wage_chg=self.max_wage_chg,
                slack=self.slack,
            )
            # Only run this after the first month.
            if self.steps > 1:
                # Firms execute planned firing.
                self.agents_by_type[Firm].do("fire")
                # Firms sets inputs and prices.
                self.agents_by_type[Firm].do(
                    "plan",
                    inventory_low=self.inventory_low,
                    inventory_high=self.inventory_high,
                    price_low=self.price_low,
                    price_high=self.price_high,
                    price_chg_prob=self.price_chg_prob,
                    max_price_chg=self.max_price_chg,
                )
                # Households search for cheaper sellers (firms).
                self.agents_by_type[Household].shuffle_do(
                    "swap_for_price",
                    swap_for_price_prob=self.swap_for_price_prob,
                    min_savings=self.min_savings,
                )
                # Households search for sellers (firms) with more inventory.
                self.agents_by_type[Household].shuffle_do(
                    "swap_for_reliability",
                    swap_for_reliability_prob=self.swap_for_reliability_prob,
                )
                # Households reset their memory of which sellers had no stock.
                self.agents_by_type[Household].do("reset_blacklist", S=self.S)
            # Households search for jobs if unemployed.
            self.agents_by_type[Household].shuffle_do(
                "unemployed_search", applys=self.applys
            )
            # Households search for higher paying jobs if underpaid.
            self.agents_by_type[Household].shuffle_do(
                "employed_search", quit_prob=self.quit_prob
            )
            # Households plan consumption.
            self.agents_by_type[Household].do(
                "budget",
                aversion=self.aversion,
                S=self.S,
                days_in_month=self.days_in_month,
            )
            # Firms reset their monthly statistics.
            self.agents_by_type[Firm].do("reset_monthly_stats")

        # Daily activities.

        # Households buy from firms to consume.
        self.agents_by_type[Household].shuffle_do("buy", browse=self.browse)
        # Firms then produce.
        self.agents_by_type[Firm].do("produce", tech_param=self.tech_param)

        # End of month activities.
        if self.end():
            # Firms pay employees (households).
            self.agents_by_type[Firm].do("pay_employees")
 
            # Run datacollector at end of month (every 21st day).
            self.datacollector.collect(self)


class Household(mesa.Agent):
    # This is a singleton household.
    def __init__(self, model):
        # Initialize the parent class with relevant parameters.
        super().__init__(model)
        # Create the agent's variables and set the initial values.
        self.employer = None
        self.most_recent_employer = None
        self.employment_hist = []
        self.sellers = []
        self.money = 0
        self.wage_r = None
        # Firms end up on blacklist if they fail to satisfy demand on any day.
        self.blacklist = []
        self.day_consume = 0
        self.paystub = 0
        
        # Axtell parameters
        self.tradeoff = random.uniform(0,1)
        self.max_effort = 1
        self.labor_connections = random.randint(2, 6)

    def initialize_money(self):
        self.money = 1e4

    def initialize_reservation(self, init_wage_r, tech_param, days_in_month):
        if init_wage_r > 0.6:
            raise Warning("Initial reservation wage might be too high.")
        self.wage_r = init_wage_r * tech_param * days_in_month
    
    # Consumption good market actions

    def assign_firms(self, S):
        # Assign seller firm relationships to households.
        all_firms = self.model.agents_by_type[Firm]
        # employer = random.choice(all_firms)
        # employer.employees.append(self)
        # self.employer = employer
        sellers = random.sample(all_firms, S)
        self.sellers = sellers
        self.blacklist = [0] * S

    def swap_for_reliability(self, swap_for_reliability_prob):
        if sum(self.blacklist) > 0:
            if incdf(swap_for_reliability_prob):
                weights = self.blacklist
                old_seller = random.choices(self.sellers, weights=weights, k=1)[0]
                all_sellers = self.model.agents_by_type[Firm]
                new_sellers = [
                    seller for seller in all_sellers if seller not in self.sellers
                ]
                new_seller = random.choice(new_sellers)
                self.sellers.remove(old_seller)
                self.sellers.append(new_seller)

    def swap_for_price(self, swap_for_price_prob, min_savings):
        if incdf(swap_for_price_prob):
            old_seller_index = random.randint(0, len(self.sellers) - 1)
            old_seller = self.sellers[old_seller_index]
            weights = [len(firm.employees) for firm in self.sellers]
            if sum(weights) > 0:
                new_seller = random.choices(self.sellers, weights=weights, k=1)[0]
                if new_seller.price / old_seller.price - 1 < -min_savings:
                    self.sellers.remove(old_seller)
                    self.sellers.append(new_seller)
                    # Update blacklist
                    del self.blacklist[old_seller_index]
                    self.blacklist.append(0)

    def reset_blacklist(self, S):
        self.blacklist = [0] * S

    def budget(self, aversion, S, days_in_month):
        total_price = sum(seller.price for seller in self.sellers)
        avg_price = total_price / S
        if self.money < 0:
            raise ValueError("A household has negative money balance.")
        if self.money > avg_price:
            self.day_consume = (self.money / avg_price) / days_in_month
        else:
            self.day_consume = pow(self.money / avg_price, aversion) / days_in_month

    def buy(self, browse):
        if self.day_consume > 0:
            self.shuffle_index = list(range(len(self.sellers)))
            consumed = 0
            firms_visited = 0
            random.shuffle(self.shuffle_index)
            for seller_index in self.shuffle_index:
                seller = self.sellers[seller_index]
                max_buyable = self.money / seller.price
                demand = min(self.day_consume - consumed, max_buyable)
                max_sellable = seller.inventory
                if demand > max_sellable:
                    quantity = max_sellable
                    self.blacklist[seller_index] = 1
                else:
                    quantity = demand
                seller.inventory -= quantity
                seller.fulfilled_demand += quantity
                consumed += quantity
                dollars_transacted = quantity * seller.price
                payment = min(dollars_transacted, self.money)
                self.money -= payment
                if self.money < 0:
                    raise TypeError(
                        f"Money is negative (${self.money}) and should not be the case."
                    )
                seller.money += payment
                firms_visited += 1
                if (
                    (consumed / self.day_consume >= 0.95)
                    or (firms_visited >= browse)
                    or (self.money == 0)
                ):

                    break

    # Labor market actions

    def initialize_singletons(self):
        # This creates a firm and appends it to the Firm AgentSet.
        Firm.create_agents(model=self.model, n=1)
        # Attach newest firm to household that created it.
        firm = self.model.agents_by_type[Firm][-1]
        self.employer = firm
        # Attach self as employee
        firm.employees.append(self)

    def optimize_effort(self, firm, return_type):
        colleagues = firm.employees.copy()
        n = len(colleagues)
        if n < 1:
            raise ValueError("This firm has less than one employee.")
        # Retrieve list of firm's employees without self.
        if self in colleagues:
            colleagues.remove(self)
        other_efforts = sum([employee.effort for employee in colleagues])

        # Define utility function
        def utility(effort):
            output = firm.constant_r * (effort + other_efforts) + pow(firm.increasing_r * (effort + other_efforts), firm.team)
        
            utility = pow(output/n, self.tradeoff) * pow(1 - effort, 1 - self.tradeoff)

            return -utility
        
        result = minimize_scalar(utility, bounds=(0, 1), method='bounded')

        if return_type == "effort":
            return result.x
        elif return_type == "utility":
            return result.fun
        else:
            raise ValueError("No return_type specified.")

    def optimize_current(self):
        self.effort = self.optimize_effort(self.employer, return_type="effort")

    def connect_labor_market(self):
        if self.labor_connections >= self.model.H:
            raise ValueError(
                "Connections exceeds total number of other households."
                )
        households = self.model.agents_by_type[Household]
        others = [household for household in households if household != self]
        self.connections = random.sample(others, self.labor_connections)

    def make_moves(self):
        if self.connections is None:
            raise ValueError(
                "Labor connections is empty."
            )
        if self.employer is None:
            raise ValueError(
                "Household does not have an employer."
            )

        # 1. Gather options in network
        firms_in_network = [friend.employer for friend in self.connections]
        options = list(set(firms_in_network))

        # 2. Add your own employer
        current = self.employer
        if current not in options:
            options.append(current)
        
        # 3. Construct a startup
        Firm.create_agents(model=self.model, n=1)
        startup = self.model.agents_by_type[Firm][-1]
        startup.employees.append(self)
        options.append(startup)

        # Loop optimize_effort.
        utilities = []
        for firm in options:
            utility = self.optimize_effort(firm, return_type="utility")
            utilities.append(utility)
        
        # Choose option with max utility.
        max_utility = max(utilities)
        best_index = utilities.index(max_utility)
        choice = options[best_index]

        # DELETE STARTUP IF NOT CHOSEN!!
        if choice != startup:
            startup.remove()

        # Assign self to new firm if appropriate.
        if (choice != current) and (choice != startup):
            choice.employees.append(self)
        
        # Remove self from old firm if appropriate.
        if choice != current:
            current.employees.remove(self)
            # Delete the firm if it has no more employees.
            if not current.employees:
                current.remove()

        # Assign choice to current employer.
        self.employer = choice

        # Update effort.
        self.effort = self.optimize_effort(choice, return_type="effort")

        # Announce choice.
        if choice == startup:
            string = "started a new firm"
        elif choice == current:
            string = "stayed with old firm"
        else:
            string = "was poached"
        print(f"Agent {self.unique_id} {string}.")

    def update_employment_hist(self):
        if self.employer is None:
            self.employment_hist.append(0)
        else:
            self.employment_hist.append(1)

    def unemployed_search(self, applys):
        if (self.employment_hist[-1] == 0) & (self.employer is None):
            firms_applied = random.sample(self.model.agents_by_type[Firm], k=applys)
            for firm in firms_applied:
                if firm.opening > 0:
                    if firm.wage > self.wage_r:
                        self.employer = firm
                        self.most_recent_employer = firm
                        firm.employees.append(self)
                        # Close the opening
                        firm.opening -= 1

                        break

    def employed_search(self, quit_prob):
        if self.employment_hist[-1] == 1:
            if self.paystub < self.wage_r:
                if incdf(quit_prob):
                    all_sellers = self.model.agents_by_type[Firm]
                    new_employers = [
                        seller
                        for seller in all_sellers
                        if seller != self.most_recent_employer
                    ]
                    new_employer = random.choice(new_employers)
                    if (new_employer.wage > self.paystub) & (new_employer.opening > 0):
                        # Remove from old employer if still employed
                        if self.employer is not None:
                            self.employer.employees.remove(self)
                        # Change household's attribute
                        self.employer = new_employer
                        self.most_recent_employer = new_employer
                        # Add to new employer
                        new_employer.employees.append(self)
                        # Close the opening
                        new_employer.opening -= 1

class Firm(mesa.Agent):
    def __init__(self, model):
        # Initialize the parent class with relevant parameters.
        super().__init__(model)
        # Create the agent's variables and set the initial values.
        """
        Starting price of a good is $1 for all firms. The most realistic interpretation here is that at the start of time, dollars were printed centrally with one dollar equal to a unit of consumption good. Each firm exchanges all initial output for dollars.
        """
        self.price = 1
        self.wage = None
        self.opening = float("inf")
        self.opening_hist = []
        self.inventory = 0
        self.employees = []
        self.fulfilled_demand = 0
        self.money = 0
        self.month_output = 0
        self.planned_firing = False
        self.retained = 0

        # Axtell parameters
        self.constant_r = random.uniform(0, 1/2)
        self.increasing_r = random.uniform(3/4, 5/4)
        self.team = random.uniform(3/2, 2)

    def initialize_price_wage(
        self,
        tech_param,
        days_in_month,
    ):
        self.price = 1
        real_output_per_capita = tech_param * days_in_month
        self.wage = max(np.random.normal(0.3, 0.1), 0) * real_output_per_capita

    def set_wage(self, max_wage_chg, slack):
        self.opening_hist.append(self.opening)
        wage_chg = random.uniform(0, max_wage_chg)
        if self.opening_hist[-1] > 0:
            self.wage = self.wage * (1 + wage_chg)

        # Check if the last x months have had no openings:
        if self.opening_hist[-slack:] == [0] * slack:
            self.wage = max(self.wage * (1 - wage_chg), 1e-9)
            if self.wage < 0:
                raise ValueError("Wage adjusted to negative value.")

    def plan(
        self,
        inventory_low,
        inventory_high,
        price_low,
        price_high,
        price_chg_prob,
        max_price_chg,
    ):
        if self.inventory < self.fulfilled_demand * inventory_low:
            self.opening = 1

            if (self.price < self.wage / 63 * price_low) & incdf(price_chg_prob):
                adjustment = random.uniform(0, max_price_chg)
                self.price = self.price * (1 + adjustment)
        if self.inventory > self.fulfilled_demand * inventory_high:
            self.opening = 0

            if len(self.employees) > 0:
                self.planned_firing = True
            if (self.price > self.wage / 63 * price_high) & incdf(price_chg_prob):
                adjustment = random.uniform(0, max_price_chg)
                self.price = max(self.price * (1 - adjustment), 1e-9)
                if self.price < 0:
                    raise ValueError("Price adjusted to negative value.")

    def fire(self):
        # Check if there are still employees, as the last one could've left before they got fired.
        if (self.planned_firing) & (len(self.employees) > 0):
            fired = random.choice(self.employees)
            self.employees.remove(fired)
            fired.employer = None
            self.planned_firing = False

    def produce(self):
        efforts = sum([employee.effort for employee in self.employees])
        output = self.constant_r * efforts + pow(self.increasing_r * efforts, self.team)
        self.inventory += output
        self.month_output += output

    def reset_monthly_stats(self):
        self.month_output = 0
        self.fulfilled_demand = 0

    def pay_employees(self):
        # Pay employees.
        if employee_count < 0:
            raise ValueError("There's no employee to pay.")
        employee_count = len(self.employees)
        wages_paid = self.money 
        total_paid = 0
        for i, employee in enumerate(self.employees):
            if i == len(self.employees) - 1:
                # Last employee gets remainder to ensure exact balance.
                payment = max(wages_paid - total_paid, 0)
            else:
                payment = wages_paid / employee_count
                total_paid += payment
            employee.money += payment
            if employee.money < 0:
                raise ValueError("Money is negative after wage pmt.")
            employee.paystub = payment
        self.money -= wages_paid