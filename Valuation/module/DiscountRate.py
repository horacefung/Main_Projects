# ---Discount Rate Calculation----#
"""
Title:      Discount Rate

Description:  There are two types of discount rates, cost of equity and cost of capital.
            In general, annualized discount rate can be thought of as what % you would've returned
            in a year for investing in something. Hence, we use it to discount future cashflows, because
            $100 dollar next year is equivalent to an amount less than $100 today after one year of growth.

            So the key question when it comes to accounting for the time value of a company's cashflow is
            what is the equivalent rate of return that receiving those CFs in the future instead of today. 
            
            For free cash-flow to equity (FCFE), we discount using the COE which is intended to reflect
            the market rate of return for equity with similar risk profile as our target company. For
            free cash-flow to firm (FCFF), we discount using the WACC (weighted average cost of capital) which
            is intended to reflect the market rate of return for a firm with similar risk profile as our target company,
            the difference being this gets averaged out with the cost of debt and is lower.

            Design Specs:
            In general, I found this easier to read top down instead of bottoms-up. What that means is the
            model will start a higher levels and work down to details, and often a higher level function will reference
            another function lower down.

Author:      Horace Fung, Nov 2021
"""

# Standard imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy

class DiscountRate():

    def __init__(self, kwargs):
        # Equity Risk Premium inputs
        #assert 

        # Beta inputs

        # Risk-free rate inputs

        return None

    # ---------------------------------- #
    # ---------------------------------- # 
    # 1) Option to choose COE or COC/WAAC workflow
    def cost_of_capital(self):
        cost_of_equity = self.cost_of_equity()
        cost_of_debt = None
        return
    
    def cost_of_equity(self):
        '''Pipeline to compute cost of equity. Steps are:
        1) Compute implied equity risk premium using an index market cap (usually S&P500)
        2) Compute weighted country risk premium
        3) Compute company's levered beta
        4) Compute risk free rate (TODO: add country level adjustment, should be minior/negligible)
        5) Compute cost of equity as: Rf + B(ERP) + SUM(Lambda_i * CRP_i)
        '''

        # Compute the risk premium as the IRR required to set S&P500 price = Dividend buyback model price
        # (1468.36, 0.0402, 0.05, 0.0402, 5)
        input_args = (self.index_cap, self.dividend_buyback_pct, self.dividend_growth, 
                    self.erp_stable_growth, self.erp_period)

        equity_risk_premium = scipy.optimize.root(fun = self.implied_erp_model, 
                                        x0=0.05,  # the initial guess
                                        args=input_args)

        # Compute country risk premium
        weighted_crp = self.weighted_crp()

        # Compute beta
        levered_beta = self.beta()

        # Compute rf
        risk_free_rate = self.risk_free_rate()

        # COE
        self.cost_of_equity = risk_free_rate + levered_beta * equity_risk_premium + weighted_crp

        return self.cost_of_equity

    # ---------------------------------- #
    # ---------------------------------- # 
    # 2) Cost of Equity calculations 
    #
    # The cost of equity is often represented by a simple CAPM model of
    # Rf + B * (ERP) + Lambda *(Country Risk Premium). This section includes all the 
    # intriciate components used to build up to this formula, and ultimately combine to a COE number.

    # --- Equity Risk Premium Section
    @staticmethod
    def implied_erp_model(irr, starting_market_price, dividend_buyback_pct, dividend_buyback_growth, stable_growth, period):
        '''Equity risk premium is suppose to reflect the risk/return of investing in the overall US market.
        Instead of looking a historical returns and computing some arbitrary average (which is backward looking as well),
        we can estimate a forward looking implied ERP by tying the current S&P500 price with a model that assumes investors
        recieve dividends/buybacks as cashflow and that amount grows over time by some growth rate.

        This function will compute the delta between the discounted sum and the actual S&P500 price. Using a root function,
        we can solve the interal rate of return required to set delta = 0, and thats the implied ERP we are looking for.
        
        Note: The implied ERP calculation based on current market cap and expected future dividends + buybacks.
        Key note is that implied ERP already includes country ERP of that nation. Only need to add country
        risk premium to other smaller parts of the business that is not reflected by the market chosen here.'''
        
        discount_sum = 0
        starting_cf = starting_market_price * dividend_buyback_pct
        cash_flows = [None] * period

        for i in range(0, period):  # Assume CF1 is received
            cash_flows[i] = starting_cf * ((1 + dividend_buyback_growth) ** (1 + i))
            
        # Perpetuity
        terminal_value = (cash_flows[-1] * (1 + stable_growth)) / ((irr - stable_growth) * (1 + irr) ** period)
        
        # NPV sum
        for i in range(0, len(cash_flows)):
            discount_sum += cash_flows[i] / (1 + irr)**(1 + i)
        
        discount_sum += terminal_value
        delta = discount_sum - starting_market_price
        
        return delta

    # --- Country Risk Premium Section
    def weighted_crp(self):
        '''Take country_risk_premium and lambda_regression functions below to
        estimate and compute a weighted CRP value. Loop through the countries
        and determine the appropriate country_risk_premiums for each country.
        Then run lambda regression to estimate their weights.
        
        Parameters:
        countries_list -- List of countries the company operates in
        soverign_rating -- List of sovereign rating of countries
        sovereign_bond -- List of sovereign bond yield
        base_bond -- The base bond yield, typically US
        sovereign_equity_vol -- List of volatilities of sovereign equity markets
        sovereign_bond_vol -- List of volatilities of sovereign bond markets

        country_returns_df -- Dataframe with company, domestic market and sovereign country equity returns
        '''

        # Compute weights
        crp_list = []
        for i in range(len(self.countries_list)):
            crp = self.country_risk_premium(sovereign_rating=self.soverign_rating[i],
                                        sovereign_bond=self.sovereign_bond[i],
                                        base_bond=self.base_bond,
                                        sovereign_equity_vol=self.sovereign_equity_vol[i],
                                        sovereign_bond_vol=self.sovereign_bond_vol[i]
                                        )
            crp_list[i] = crp
        
        crp_df = pd.DataFrame({'countries':self.countries_list, 'crp':crp_list})

        # Estimate coefficients
        lambda_df, adj_r2 = self.lambda_regression(self.returns_df, self.countries_list)

        # Compute weighted CRP
        crp_df = pd.merge(crp_df, lambda_df, how='left', on='countries')
        self.weighted_crp = np.sum(crp_df['crp'] * crp_df['lambda'])

        return self.weighted_crp              

    @staticmethod
    def country_risk_premium(kwargs):
        '''Choose from 3 options to compute a region's CRP, depending on
        what the country risk rating is or if its even available.
        1) If sovereign_ratin is triple A, no country risk premium
        2) Any other ratings, use the default spread and convert with an equity-to-debt volatility ratio
        3) Use the vendor Political Risk Services group's data. '''

        if kwargs['sovereign_rating'] == 'AAA':
            return 0.0
        elif kwargs['sovereign_rating'] != None:
            default_spread = kwargs['sovereign_bond'] - kwargs['base_bond']
            return default_spread * (kwargs['sovereign_equity_vol'] / kwargs['sovereign_bond_vol'])
        #elif kwargs['prs_risk_estimate'] != None:
        #    return kwargs['prs_risk_estimate']
        else:
            return print('Missing input')
    
    @staticmethod
    def lambda_regression(returns_df, countries_list):
        ''' Linear regression to estimate lambda coefficients on CRP. This is a regression
        of the country's bond returns versus the target company's equity return. The 
        design idea is to by default include a static & broad range of countries in the df,
        then filter to the relevant ones using countries_list.
        
        Keyword Arguments:
        returns_df - A dataframe storing the company's return and relevant country bond returns
        countries_list - A list of countries to include. Will match column names.
        '''

        assert 'company_returns' in returns_df.columns == True, 'Missing target company returns'
        assert 'market_returns' in returns_df.columsn == True, 'Missing market returns'
        assert len(countries_list) > 0, 'No countries specified'
        premium_predictors = ['market_returns'] + countries_list  # full predictor set

        # Regression
        X = returns_df['company_returns']
        y = returns_df[premium_predictors]

        regressor = LinearRegression(X, y)
        y_hat = regressor.predict(X)
        r2 = r2_score(y, y_hat)
        adj_r2 = 1-(1-r2)*(len(X)-1)/(len(X)-len(X.columns)-1)
        
        # Get country coefficients
        lambda_df = pd.DataFrame({'countries' : premium_predictors, 'lambda':regressor.coef_})
        lambda_df = lambda_df[lambda_df['countries'] != 'market_returns']

        return lambda_df, adj_r2

    # --- Beta Section
    def beta(self):
        '''Compute a bottoms-up beta for different industries the target business operates
        in. For each industry, unlever the industry average leverage and then lever back to
        the target firm's financial leverage. Afterwards, compute a weighted average beta
        using the EV/Sales x Firm Revenue (approximating the firm's Enterprise Value for each
        business stream, aka equity + debt) as weights.

        Parameters
        industry_list -- List of industries
        industry_betas -- List of average regression betas for each industry
        industry_fc_vc_ratios -- List of avg fixed/variable cost ratios for each industry
        debt_beta -- The debt beta of the firm, usually unavailable
        debt_to_equity_ratio -- The debt/equity ratio of the firm
        tax_rate -- The standard firm tax rate
        industry_ev_sales -- List of industry EV/Sales ratios
        revenues -- List of the firms revenue broken by industry
        '''
        
        # Compute levered betas for each industry the company is in
        levered_betas = []
        for i in self.industry_list:
            # Unlever the overall industry pure business beta (from some regression)
            # by accounting for the industry average fixed-to-variable cost ratio
            unlevered_beta = self.industry_betas[i] * (1 + self.industry_fc_vc_ratios[i])
            # Hamada's equation: adjust back for firm specific leverage (using financial leverage)
            # effectively the unlevered_beta + some extra risk from leverage
            levered_beta = unlevered_beta * (1 + self.debt_to_equity_ratio * (1 - self.tax_rate))

            if self.debt_beta is not None:
                # Hard to find, but if user knows the debt market risk
                levered_beta = levered_beta - (self.debt_betas * self.debt_to_equity_ratios * (1 - self.tax_rate))
            else:
                # Usually just assume neglibile
                None

            levered_betas[i] = levered_beta

        # Compute a revenue weighted average. Dataframe easier to debug & examine.
        beta_df = pd.DataFrame({'industry':self.industry_list, 
                                'industry_ev_sales':self.industry_ev_sales,
                                'revenue':self.revenues,
                                'beta':levered_betas})

        beta_df['value'] = beta_df['revenue'] * beta_df['industry_ev_sales']
        beta_df['weight'] = beta_df['value'] / np.sum(beta_df['value'])
        self.beta = np.sum(beta_df['weight'] * beta_df['beta'])

        return self.beta


    # --- Risk-free Rate Section (Let's assume no global variation for now)
    def risk_free_rate(self):
        return self.risk_free_rate

    
if __name__ == '__main__':
    test = DiscountRate()
    test.cost_of_debt()