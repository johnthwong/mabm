# Has multi-dimensional arrays and matrices.
# Has a large collection of mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd

# Data visualization tools.
import seaborn as sns
import mesa
import random


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

# Functions for computing model-level moments
def compute_employment(model):
    employment_vector = [
        employment(a.employer) for a in model.agents_by_type[Household]
        ]
    return sum(employment_vector)


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
        slack: int = 24,
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
        
        # Create agents.
        Household.create_agents(model=self, n=self.H)
        Firm.create_agents(model=self, n=self.F)
        
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
            # `compute_employment` is the helper function defined above. Passed not as a string.
            model_reporters={"employment": compute_employment},
            agent_reporters={"money": "money"},
            agenttype_reporters={
                Household: {
                    "employment": lambda a: employment(a.employer),
                    "full_demand": lambda a: a.month_consume,
                },
                Firm: {
                    "output": lambda a: a.month_output,
                    "price": lambda a: a.price,
                    "employees": lambda a: len(a.employees),
                    "fulfilled_demand": lambda a: a.fulfilled_demand,
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
    def step(self):
        # Start of month activities.
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
       
        if self.steps > 1:
            # Households buy from firms to consume.
            self.agents_by_type[Household].shuffle_do("buy", browse=self.browse)
        
        # Firms then produce.
        self.agents_by_type[Firm].do("produce", tech_param=self.tech_param)
        
        # End of month activities.
        if self.end():
            
            # Firms pay employees (households).
            self.agents_by_type[Firm].do("pay_employees", buffer=self.buffer)
            
            # Firms pay shareholders (households).
            self.agents_by_type[Firm].do("pay_shareholders")
            
            # Households update their reservation wage.
            self.agents_by_type[Household].do(
                "adjust_reservation", lower_wage_r=self.lower_wage_r
            )
            
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
                new_firm_weights = [len(firm.employees) for firm in new_sellers]
                new_seller = random.choices(new_sellers, weights=new_firm_weights, k=1)[0]
                self.sellers.remove(old_seller)
                self.sellers.append(new_seller)
                if len(self.sellers) != len(set(self.sellers)):
                    raise ValueError("Sellers list has duplicates.")

    def swap_for_price(self, swap_for_price_prob, min_savings):
        if incdf(swap_for_price_prob):
            old_seller_index = random.randint(0, len(self.sellers) - 1)
            old_seller = self.sellers[old_seller_index]
            all_sellers = self.model.agents_by_type[Firm]
            new_sellers = [
                seller for seller in all_sellers if seller not in self.sellers
            ]
            weights = [len(firm.employees) for firm in new_sellers]
            if sum(weights) > 0:
                new_seller = random.choices(new_sellers, weights=weights, k=1)[0]
                if new_seller.price / old_seller.price - 1 <= -min_savings:
                    self.sellers.remove(old_seller)
                    self.sellers.append(new_seller)
                    # Update blacklist
                    del self.blacklist[old_seller_index]
                    self.blacklist.append(0)
                    if len(self.sellers) != len(set(self.sellers)):
                        raise ValueError("Sellers list has duplicates.")

    def reset_blacklist(self, S):
        self.blacklist = [0] * S

    def budget(self, aversion, S, days_in_month):
        total_price = sum(seller.price for seller in self.sellers)
        avg_price = total_price / S
        if self.money < 0:
            raise ValueError("A household has negative money balance.")
        if self.money > avg_price:
            self.month_consume = (self.money / avg_price)
        else:
            self.month_consume = pow(self.money / avg_price, aversion)
        self.day_consume = self.month_consume/days_in_month

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
                seller.full_demand += demand
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

    def adjust_reservation(self, lower_wage_r):
        if (self.employer is not None) & (self.paystub > self.wage_r):
            self.wage_r = self.paystub

        if self.employer is None:
            self.wage_r = (1 - lower_wage_r) * self.wage_r


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
        self.full_demand = 0
        self.money = 0
        self.month_output = 0
        self.planned_firing = False
        self.retained = 0

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

    def produce(self, tech_param: float, tech_type="linear"):
        if tech_type == "linear":
            output = tech_param * len(self.employees)
        self.inventory += output
        self.month_output += output

    def reset_monthly_stats(self):
        self.month_output = 0
        self.fulfilled_demand = 0
        self.full_demand = 0

    def pay_employees(self, buffer):
        # Pay employees.
        employee_count = len(self.employees)
        wage_bill = employee_count * self.wage
        if employee_count > 0:
            wages_paid = min(wage_bill, self.money)
            total_paid = 0
            for i, employee in enumerate(self.employees):
                if i == len(self.employees) - 1:
                    # Last employee gets remainder to ensure exact balance
                    payment = max(wages_paid - total_paid, 0)
                else:
                    payment = wages_paid / employee_count
                    total_paid += payment
                employee.money += payment
                if employee.money < 0:
                    raise ValueError("Money is negative after wage pmt.")
                employee.paystub = payment
            self.money -= wages_paid
        self.retained = min(wage_bill * buffer, self.money)

    def pay_shareholders(self):
        # Pay shareholders.
        dividends = max(self.money - self.retained, 0)
        if dividends > 0:
            owners = self.model.agents_by_type[Household]
            equities = [owner.money for owner in owners]
            equity = sum(equities)
            total_distributed = 0
            if equity > 0:
                for i, shareholder in enumerate(owners):
                    if i == len(owners) - 1:
                        # Last shareholder gets remainder to ensure exact balance
                        share_amount = max(dividends - total_distributed, 0)
                    else:
                        share = shareholder.money / equity
                        share_amount = share * dividends
                        total_distributed += share_amount
                    shareholder.money += share_amount
                    if shareholder.money < 0:
                        raise ValueError("Money is negative after dividend pmt.")
        self.money -= dividends
