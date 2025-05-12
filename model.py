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
    def __init__(
            self, 
            seed:int = None,
            H:int = 1000,
            S:int = 7,
            max_wage_chg = 0.019,
            slack:int = 6,
            inventory_low = 0.25,
            inventory_high = 1,
            price_low = 1.025,
            price_high = 1.15,
            price_chg_prob = 0.75,
            max_price_chg = 0.02,
            tech_param:int = 3,
            swap_for_reliability_prob = 0.25,
            swap_for_price_prob = 0.25,
            min_savings:float = 0.01,
            applys = 5,
            quit_prob = 0.1,
            aversion = 0.9,
            browse = 7,
            buffer:int = 0.1,
            init_wage_r = 0.25,
            lower_wage_r = 0.1,
            ):
        # Initialize the parent class with relevant parameters.
        super().__init__(seed=seed)
        # Parameters.
        self.months = 0
        self.days_in_month = 21
        self.H = H
        self.F = int(H*1e-1) # Number of firms
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
        # Debug
        self.counter = 0
        # Create agents.
        Household.create_agents(model=self, n=self.H)
        Firm.create_agents(model=self, n=self.F)
        # Pair households to firms.
        print("Running: assign_firms")
        self.agents_by_type[Household].shuffle_do("assign_firms", S = self.S)
        print(f"Counter: {self.counter}")
        if self.counter < self.H:
            raise ValueError("Counter too low.")
        self.counter = 0
        
        # Initialize money.
        print("Running: initialize_money")
        self.agents_by_type[Household].do(
            "initialize_money",
            tech_param = self.tech_param, 
            days_in_month = self.days_in_month,
            )
        print(f"Counter: {self.counter}")
        if self.counter < self.H:
            raise ValueError("Counter too low.")
        self.counter = 0
        
        # Initialize price and wages.
        print("Running: initialize_price_wage")
        self.agents_by_type[Firm].do(
            "initialize_price_wage",
            tech_param = self.tech_param, 
            days_in_month = self.days_in_month, 
            price_low = self.price_low, 
            price_high = self.price_high,
            )
        print(f"Counter: {self.counter}")
        if self.counter < self.F:
            raise ValueError("Counter too low.")
        self.counter = 0
        
        # Initialize reservation wage.
        print("Running: initialize_reservation")
        self.agents_by_type[Household].do(
            "initialize_reservation",
            init_wage_r = self.init_wage_r,
            tech_param=self.tech_param,
            days_in_month=self.days_in_month
            )
        print(f"Counter: {self.counter}")
        if self.counter < self.H:
            raise ValueError("Counter too low.")
        self.counter = 0
        
        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            agent_reporters={'money':'money'},
            agenttype_reporters={
                Household:{
                    "employment":lambda a: employment(a.employer),
                },
                Firm:{
                    "output":lambda a: a.month_output,
                    "price":lambda a: a.price,
                    "employees": lambda a: len(a.employees),
                    "demand": lambda a: a.fulfilled_demand,
                    'vacancies': lambda a: a.opening,
                    'inventory': lambda a: a.inventory,
                }
            }
        )
    def end(self):
        if self.steps % self.days_in_month == 0:
            return True
    def start(self):
        if (self.steps % self.days_in_month == 1):
            return True
    def record_month(self):
        self.months += 1
    def step(self):
        # Start of month activities.
        print(f"Today is day {self.steps}.")
        if self.start():
            print("start of month.")

            self.record_month()
        
            # Households recall whether they were employed last month.
            print("Running: update_employment_hist")
            self.agents_by_type[Household].do("update_employment_hist")
            print(f"Counter: {self.counter}")
            if self.counter < self.H:
                raise ValueError("Counter too low.")
            self.counter = 0
            
            print("Running: set_wage")
            self.agents_by_type[Firm].do(
                "set_wage", 
                max_wage_chg = self.max_wage_chg,
                slack = self.slack,
                )
            print(f"Counter: {self.counter}")
            self.counter = 0
            
            if self.steps > 1:

                print("Running: fire")
                self.agents_by_type[Firm].do("fire")
                print(f"Counter: {self.counter}")
                self.counter = 0
            
                print("Running: plan")
                self.agents_by_type[Firm].do(
                    "plan", 
                    inventory_low = self.inventory_low,
                    inventory_high = self.inventory_high,
                    price_low = self.price_low,
                    price_high = self.price_high,
                    price_chg_prob = self.price_chg_prob,
                    max_price_chg = self.max_price_chg,
                    )
                print(f"Counter: {self.counter}")
                self.counter = 0
            
                print("Running: swap_for_reliability")
                self.agents_by_type[Household].shuffle_do(
                    "swap_for_reliability",
                    swap_for_reliability_prob = self.swap_for_reliability_prob,
                )
                print(f"Counter: {self.counter}")
                self.counter = 0
                
                print("Running: swap_for_price")
                self.agents_by_type[Household].shuffle_do(
                    "swap_for_price",
                    swap_for_price_prob = self.swap_for_price_prob,
                    min_savings = self.min_savings,
                )
                print(f"Counter: {self.counter}")
                self.counter = 0
                
                print("Running: reset_blacklist")
                self.agents_by_type[Household].do(
                    "reset_blacklist", S = self.S
                    )
                print(f"Counter: {self.counter}")
                self.counter = 0
                
            print("Running: unemployed_search")
            self.agents_by_type[Household].shuffle_do(
                "unemployed_search", 
                applys = self.applys
                )
            print(f"Counter: {self.counter}")
            self.counter = 0
            
            print("Running: employed_search")
            self.agents_by_type[Household].shuffle_do(
                "employed_search", 
                quit_prob = self.quit_prob
                )
            print(f"Counter: {self.counter}")
            self.counter = 0
            
            print("Running: budget")
            self.agents_by_type[Household].do(
                "budget", 
                aversion=self.aversion, 
                S=self.S,
                days_in_month=self.days_in_month
                )
            print(f"Counter: {self.counter}")
            self.counter = 0
            
            print("Running: reset_monthly_stats")
            self.agents_by_type[Firm].do("reset_monthly_stats")
            print(f"Counter: {self.counter}")
            self.counter = 0
            
        # Daily activities.
        print("Running: buy")
        self.agents_by_type[Household].shuffle_do(
            "buy", browse=self.browse
            )
        print(f"Counter: {self.counter}")
        if self.counter == 0:
            raise ValueError("Counter too low.")
        self.counter = 0
        
        print("Running: produce")
        self.agents_by_type[Firm].do("produce", tech_param=self.tech_param)
        print(f"Counter: {self.counter}")
        if self.counter == 0:
            raise ValueError("Counter too low.")
        self.counter = 0
        
        # End of month activities.
        if self.end():
            print("Running: do_finances")
            self.agents_by_type[Firm].do("do_finances", buffer=self.buffer)
            print(f"Counter: {self.counter}")
            self.counter = 0
            
            print("Running: adjust_reservation")
            self.agents_by_type[Household].do(
                "adjust_reservation", lower_wage_r = self.lower_wage_r
                )
            print(f"Counter: {self.counter}")
            self.counter = 0
            
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
    def initialize_money(self, tech_param, days_in_month):
        '''
        This model is extremely sensitive to initial nominal conditions.
        There are three variables to deliberately initialize: price, money, and wage.
        Let us assume that each dollar initially represents one unit of consumption good. This still allows the nominal to diverge from the real emergently, but we just assume the two are the same for now. This exogenous assumption can imply stable values for the other two.
        We know that each worker produces 3*21 goods---and that this needs to somewhat clear. Thus, each worker should have 63 dollars. Call this their endowed wealth.
        We know that price should be a function of 'marginal cost', which is not defined in the paper. The marginal cost of 63 goods is the wage, so let us assume the marginal cost of one good is wage/63. We know that 63*price (which is 1)/price_low is the ceiling wage; if divided by price_high, it's the floor wage. So we take the average of the two and use that as the denominator.
        We can add noise to each agent's attributes, but the mean should be this. 
        '''
        real_output_per_capita = tech_param * days_in_month
        self.money = np.random.normal(
            100, 
            100*1e-1
            )
        self.model.counter += 1
    def initialize_reservation(self, init_wage_r, tech_param, days_in_month):
        if init_wage_r > 0.6:
            raise Warning("Initial reservation wage might be too high.")
        self.wage_r = init_wage_r * tech_param * days_in_month
        self.model.counter += 1
    def assign_firms(self, S):
        # Assign seller firm relationships to households.
        all_firms = self.model.agents_by_type[Firm]
        # employer = random.choice(all_firms)
        # employer.employees.append(self)
        # self.employer = employer
        sellers = random.sample(all_firms, S)
        self.sellers = sellers
        self.blacklist = [0]*S
        self.model.counter += 1
    def update_employment_hist(self):
        if self.employer is None:
            self.employment_hist.append(0)
        else:
            self.employment_hist.append(1)
        self.model.counter += 1
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
                self.model.counter += 1
    def swap_for_price(self, swap_for_price_prob, min_savings):
        if incdf(swap_for_price_prob):
            old_seller = random.choice(self.sellers)
            weights = [len(firm.employees) for firm in self.sellers]
            if sum(weights) > 0:
                new_seller = random.choices(
                    self.sellers, weights=weights, k=1
                    )[0]
                if new_seller.price/old_seller.price - 1 < - min_savings:
                    self.sellers.remove(old_seller)
                    self.sellers.append(new_seller)
                    self.model.counter += 1
    def reset_blacklist(self, S):
        self.blacklist = [0]*S
        self.model.counter += 1
    def unemployed_search(self, applys):
        if self.employment_hist[-1] == 0:
            firms_applied = random.sample(
                self.model.agents_by_type[Firm],
                k=applys
                )
            for firm in firms_applied:
                if (firm.wage > self.wage_r) & (firm.opening > 0):
                    self.employer = firm
                    self.most_recent_employer = firm
                    firm.employees.append(self)
                    # Close the opening
                    firm.opening -= 1
                    self.model.counter += 1
                    break
    def employed_search(self, quit_prob):
        if self.employment_hist[-1] == 1:
            if self.paystub < self.wage_r:
                if incdf(quit_prob):
                    all_sellers = self.model.agents_by_type[Firm]
                    new_employers = [
                        seller for seller in all_sellers if seller != self.most_recent_employer
                    ]
                    new_employer = random.choice(new_employers)
                    if (new_employer.wage > self.paystub) & (new_employer.opening > 0):
                        self.employer = new_employer
                        self.most_recent_employer = new_employer
                        new_employer.employees.append(self)
                        # Close the opening
                        new_employer.opening -= 1
                        self.model.counter += 1
    def budget(self, aversion, S, days_in_month):
        total_price = sum(seller.price for seller in self.sellers)
        avg_price = total_price/S
        if self.money > avg_price:
            self.day_consume = (self.money/avg_price)/days_in_month
        else:
            self.day_consume = pow(self.money/avg_price, aversion)/days_in_month
        self.model.counter += 1
    def buy(self, browse):
        if self.day_consume > 0:
            self.shuffle_index = list(range(len(self.sellers)))
            consumed = 0
            firms_visited = 0
            random.shuffle(self.shuffle_index)
            for seller_index in self.shuffle_index:
                seller = self.sellers[seller_index]
                max_buyable = self.money/seller.price
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
                self.money -= min(dollars_transacted, self.money)
                if self.money < 0:
                    raise TypeError(
                        f"Money is negative (${self.money}) and should not be the case."
                        )
                seller.money += dollars_transacted
                firms_visited += 1
                if( 
                    (consumed/self.day_consume >= 0.95) 
                    or (firms_visited >= browse)
                    or (self.money == 0)
                ):
                    self.model.counter += 1
                    break
    def adjust_reservation(self, lower_wage_r):
        if (self.employment_hist[-1] == 1) & (self.paystub > self.wage_r):
            self.wage_r = self.paystub
            self.model.counter += 1
        if (self.employment_hist[-1] == 0):
            self.wage_r = (1 - lower_wage_r) * self.wage_r
            self.model.counter += 1
class Firm(mesa.Agent):
    def __init__(self, model):
        # Initialize the parent class with relevant parameters.
        super().__init__(model)
        # Create the agent's variables and set the initial values.
        '''
        Starting price of a good is $1 for all firms. The most realistic interpretation here is that at the start of time, dollars were printed centrally with one dollar equal to a unit of consumption good. Each firm exchanges all initial output for dollars.
        '''
        self.price = 1 
        self.wage = None
        self.opening = float('inf')
        self.opening_hist = []
        self.inventory = 0
        self.employees = []
        self.fulfilled_demand = 0
        self.money = 0
        self.month_output = 0
        self.planned_firing = False
    def initialize_price_wage(self, tech_param, days_in_month, price_low, price_high):
        # See Household.initialize_money() for explanation.
        self.price=1
        real_output_per_capita = tech_param * days_in_month
        price_per_mc = random.uniform(1, 5)
        self.wage = self.price * real_output_per_capita / price_per_mc
        self.model.counter += 1
    def set_wage(self, max_wage_chg, slack):
        self.opening_hist.append(self.opening)
        wage_chg = random.uniform(0, max_wage_chg)
        if self.opening_hist[-1] > 0:
            self.wage = self.wage*(1 + wage_chg)
            self.model.counter += 1
        # Check if the last x months have had no openings:
        if self.opening_hist[-slack:] == [0]*slack:
            self.wage = self.wage*(1 - wage_chg)
            self.model.counter += 1
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
            self.model.counter += 1
            if (self.price < self.wage * price_low) & incdf(price_chg_prob):
                adjustment = random.uniform(0, max_price_chg)
                self.price = self.price * (1 + adjustment)
        if self.inventory > self.fulfilled_demand * inventory_high:
            self.opening = False
            self.model.counter += 1
            if len(self.employees) > 0:
                self.planned_firing = True
            if (self.price > self.wage * price_high) & incdf(price_chg_prob):
                adjustment = random.uniform(0, max_price_chg)
                self.price = self.price * (1 - adjustment)
    def fire(self):
        if self.planned_firing:
            fired = random.choice(self.employees)
            self.employees.remove(fired)
            fired.employer = None
            self.planned_firing = False
    def produce(self, tech_param:float, tech_type="linear"):
        if tech_type == "linear":
            output = tech_param * len(self.employees)
        self.inventory += output
        self.month_output += output
        if self.month_output > 0:
            self.model.counter += 1
    def reset_monthly_stats(self):
        self.month_output = 0
        self.fulfilled_demand = 0
        self.model.counter += 1
    def do_finances(self, buffer, ownership="sovereign"):
        '''
        Firms do three things. 
        1. Pay employees.
        2. Pay shareholders.
        3. The rest is retained profits.
        '''
        # Pay employees.
        employee_count = len(self.employees)
        wage_bill = employee_count * self.wage
        if employee_count > 0:
            wages_paid = min(wage_bill, self.money)
            payment = wages_paid/employee_count
            for employee in self.employees:
                employee.money += payment
                employee.paystub = payment
        # Pay shareholders.
        dividends = max(self.money - (1 + buffer) * wage_bill, 0)
        if dividends > 0:
            if ownership == "sovereign":
                owners = self.model.agents_by_type[Household]
                equities = [owner.money for owner in owners]
                equity = sum(equities)
                if equity > 0:
                    for shareholder in owners:
                        share = shareholder.money/equity
                        shareholder.money += share * dividends
        self.money -= dividends
        self.model.counter += 1