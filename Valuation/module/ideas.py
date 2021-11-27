# ---Ideas----#
"""
Title:       Scratch Script

Description: Scratch script for ideas on various functions we need to build out
             equity valuation module.

Author:      Horace Fung, Dec 2020
"""

# import packages
import pickle
from sklearn.externals import joblib
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy
warnings.filterwarnings('ignore')


# Idea functions

# --- Usesable functions --- #
def default_spread(sovereign_bond_base, base_bond):

	default_spread = sovereign_bond_base - base_bond

	return default_spread

# --- Risk-free Rate --- #
def rf_country_adj(method, sovereign_bond_orig, sovereign_bond_base, base_bond, default_spread,
	forward_rate, spot_rate, base_maturity, real_rf, nominal_rf):

	'''Adjusting risk-free rate for country risk differences. Four options:
	1. Use or estimate a default_spread for calculating rf
	2. Use currency spot and forward rates to base for calculating rf
	3. Use real rf
	4. Use nominal rf'''

	if method == 'default_spread':
		sovereign_rf = sovereign_bond_orig - default_spread

	elif method == 'forward_rates':
		sovereign_rf = ((forward_rate * (1 + base_bond)**base_maturity)/spot_rate)**(1/base_maturity) - 1

	elif method == 'real_rf':
		sovereign_rf = real_rf

	elif method == 'nominal_rf':
		sovereign_rf = nominal_rf

	else:
		return print('Please specify method')

	return sovereign_rf



# --- Equity Risk Premium --- (Some Corporate RP + some country RP) #
def implied_erp(irr, starting_market_price, dividend_buyback_pct, dividend_buyback_growth, stable_growth, period):

	'''The implied ERP calculation based on current market cap and expected future dividends + buybacks.
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
    output = discount_sum - starting_market_price
    
    return output


 root(fun = implied_erp, x0=0.05, args=(1468.36, 0.0402, 0.05, 0.0402, 5))


def country_risk_premium(sovereign_rating, default_spread, prs_risk_estimate,
	sovereign_equity_vol, sovereign_bond_vol):

	'''Choose from 3 options to compute a region's CRP, depending on
	what the country risk rating is or if its even available'''

	if sovereign_rating == 'AAA':
		crp = 0

	elif sovereign_rating != None:
		crp = default_spread * (sovereign_equity_vol / sovereign_bond_vol)

	elif prs_risk_estimate != None:
		crp = prs_risk_estimate

	else:
		return print('Missing input')

	return crp

# There are other approaches to estimating the CRP coefficients, but
# a regression is the most robust and data driven method.
def lambda_regression(returns_df):

	''' Linear regression to estimate lambda coefficients on country risk.
	y variable is the company's return, X1 is base equity risk premium and
	X1+n where n > 0 are the country risk premiums

	returns_df structure is:
	1. company returns
	2. market equity risk premium
	3. list of columns of country risk premiums to regress on'''

	# Check the column names and order are correct
	if returns_df.columns[0] != 'company_returns':
		print('First column should be company_returns')

	elif returns_df.columns[1] != 'market_erp':
    	print('Second column should be market_erp')

	else:
		None

	# Regression
	X = returns_df.iloc[:, 1:]
	y = returns_df.iloc[:, 0]

	regressor = LinearRegression(X, y)
	y_hat = regressor.predict(X)
	r2 = r2_score(y, y_hat)
	adj_r2 = 1-(1-r2)*(len(X)-1)/(len(X)-len(X.columns)-1)
	
	# Get country coefficients
	lambda_df = pd.DataFrame({'premiums' : list(X.columns), 'lambda':regressor.coef_})

	return lambda_df, adj_r2

 
def weighted_crp(df):

	'''Take a dataframe of countries/regions, their CRP and the weights.
	Pretty minor step, to the point where we may not have this as an individual
	function, but we'll see'''

	df['weighted_crp'] = df['crp'] * df['coefficients']

	return df['weighted_crp'].sum()


# ---- Beta Calculation / Manipulations ------ #
# This concludes cost of equity calculations. 
# We have Rf, CRP, Market Risk Premium and Beta.

def unlever_beta(levered_beta, fc_vc_ratio):

	'''Unlever beta calculation, can be used to unlever pure business beta with
	business average fc_vc ratio to get the unlever beta.'''

	unlever_b = levered_beta * (1 + fc_vc_ratio)

	return unlever_b

def lever_beta(unlevered_beta, debt_to_equity_ratio, tax_rate, debt_beta):

	'''Lever beta calculation, can be use to take unlevered beta and lever
	up to firm specific financial leverage.'''

	levered_b = unlevered_beta * (1 + debt_to_equity_ratio * (1 - tax_rate))
	levered_b = levered_b - (debt_beta * debt_to_equity_ratio * (1 - tax_rate))

	return levered_b


# No specific cost of debt calculations, cost of debt is fairly straightforward
# ---- Cost of Capital ---- #
def wacc_calc(cost_of_equity, cost_of_debt, equity, debt, tax_urate, real = False,
	inflation_sovereign = None, debt_component = None):

	equity_component = (equity / (equity + debt)) * cost_of_equity
	debt_component = (debt / (equity + debt)) * cost_of_debt * (1 - tax_rate)
	wacc = equity_component + debt_component

	if real == True:
		adjustment = (1 + inflation_sovereign) / (1 + inflation_base)
		wacc = (1 + wacc) * adjustment

	return wacc

# This is to make sure our E, D ratios account for all equity and debt
def hybrid_securities(principal, bond_yield, maturity, interest_rate, market_value):

	''' For hybrid securities like convertible bonds, we should
	break them down to debt and equity portion. Debt will be the defined
	value based on maturity, yield and principal. Equity will be the
	difference between the calculated debt and market value of these
	securities.'''

	discounted_sum = 0

    for i in range(0, maturity): 
        discounted_sum += (principal * bond_yield) / ((1 + interest_rate) ** (i))

    discounted_principal = principal / (1 + interest_rate) ** maturity

    debt_portion = discounted_sum + discounted_principal
    equity_portion = market_value - debt_portion
    output = {'debt_portion' : debt_portion, 'equity_portion' : equity_portion}

    return output


# ---- Estimating Cash Flows ----- #











