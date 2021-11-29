# --- Free Cashflow Calculation----#
"""
Title:      Free Cashflows

Description: There are twom  types of free cash flows, FCFE (free cash flow to equity) and 
            FCFF (free cash flow to firm). Generally, for equity valuation we prefer to use
            FCFE since its directly calculating the cashflows to equity. What's the difference?
            FCFE is smaller, its the free cash that flows to equity holders after paying out
            debt holders.

            Adjustments: 
            
            1) The accounting rules in financial reporting doesn't always classified
            expenses in a way that truly reflects the business operations and capital investments.
            In general, we care about 3 types-- operating, financial and capital expense.
            Operating: Expense a business incurs from normal operations
            Financial: A tax deductible commitment regardless of operations
            Capital: An expense taht is expected to generate benefits over multiple periods

            2) 12-months trailing, 10Qs are released quarterly and 10Ks are released annually.
            We can update the financials by adding any new 10Qs to a 10K and subtracting 
            previous 10Qs. E.g. 2021_Q2 + 2021_Q1 + 2020_10K - 2020_Q3 - 2020_Q2.


Author:      Horace Fung, Nov 2021
"""

# Standard imports
from os import stat
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy

# Project/helper imports
import helper

class FreeCashFlows():

    def __init__():
        None

    # ---------------------------------- #
    # ---------------------------------- # 
    # 1) Option to choose FCFE, Dividends or FCFF
    def fcfe():
        # net income
        # - capex
        # + depreciation
        # - change in non-cash working capital
        # - net debt
        # - preferred dividend
        # - interest income from cash
        return None
    
    def fcff():
        # ebit(1-t)
        # - capex
        # + depreciation
        # - change in non-cash working capital
        return None
    
    def dividends():
        return None

    # ---------------------------------- #
    # ---------------------------------- # 
    # 2) Adjustments (re-categorizing expenses + impact to other statements)
    @staticmethod
    def adj_operating_lease(op_lease_dict, cost_of_debt):
        ''' An operating lease is a contract that allows the company to use
        an asset (e.g. machinery) without owning the asset, and instead pay
        some fixed payments. While it is called "operating", it behaves more like
        financial expenses due to the commital nature. 
        
        Hence, we should treat it like debt payments and convert the 
        operating lease into debt, then shift the expense from EBIT down 
        to financial expense (Net Income remains the same). Then add the debt to
        balance sheet (and subsequently debt ratio calculations).
        '''

        # Convert lease commitments to debt (balance sheet)
        op_debt = helper.pv_specific_cf(op_lease_dict['commitments'], 
                                        cost_of_debt,
                                        0) # typically no principal on operating lease

        # Compute operating income adjustments (income/cash statement)
        adj_op_expense = op_lease_dict['curr_op_expense']  # add back orig expense
        adj_depreciation = -op_debt / len(op_lease_dict['commitments'])  # subtract new depreciation

        return op_debt, adj_op_expense, adj_depreciation


    # 3) Captialize R&D Expenses
    @staticmethod
    def capitalize_rd(rd_dict):
        ''' R&D expenses are categorized as financial expenses. But in this 
        context, they are generating an asset that provide future growth and cashflows.
        Hence, it's more logical to categorize them as capital expenditures. This means
        1) adding back the original R&D expense 2) subtracting a new & smaller ammortization
        expense 3) increased book asset & equity and 4) increase in CAPEX. 

        Ammortization refers to spreading out capital expense on an intangible asset over time.
        Part of the research (intangible) is ammortized because research doesn't have perfect
        $ translation to the asset it will create. We look at past R&D expenses plus a straight
        line ammortization schedule to figure out the value of the asset created today. 

        Dictionary Arguments:
        rd_expense -- List of past rd expenses (from 10Ks), in descending order [0] = current year
        amortizable_life -- The life (2-10 years) of the intangible R&D asset
        '''

        assert len(rd_dict['rd_expense']) == rd_dict['amortizable_life'], 'Different R&D expenses than amortiziable life'

        # Create amortization schedule
        increments = 1/rd_dict['amortizable_life']
        schedule = list(range(0, rd_dict['amortizable_life'] + 1))
        schedule = [1 - (increments * i) for i in schedule]

        # 


        return None


    