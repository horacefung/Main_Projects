# ---General Calculations----#
"""
Title:      Common helper functions (e.g. PV calculations)

Description: 

Author:      Horace Fung, Nov 2021
"""

def annuity_pv(payment, rate, periods):
    '''Compute the present value of an annuity. An annuity
    is a series of regular payments across a period of time,
    no principal at the end.

    Starts at t=1, eg 4 periods would be t=(1,2,3,4)
    '''
    annuity = payment * (1 - (1 / (1 + rate)**periods)) / rate

    return annuity

def annuity_principal_pv(payment, rate, periods, principal):
    ''' Same as above but with principal payment at the end.
    '''
    annuity = annuity_pv(payment, rate, periods)
    principal = principal / (1 + rate)**periods

    return annuity + principal

def pv_specific_cf(cf_array, rate, principal):
    '''Present value calculation with specific array of cashflows
    and constant discount rate.
    '''
    periods = len(cf_array)
    pv = 0
    for i in range(0, periods):
        discount_rate = (1 + rate)**(i + 1)
        pv += cf_array[i] / discount_rate
    
    pv += principal / (1 + rate)**periods

    return pv

if __name__ == '__main__':
    #print(annuity_pv(100, 0.05, 6))
    print(pv_specific_cf([899, 846, 738, 598, 477, 982.5, 982.5], 0.06, 0))