"""
Climate metabolism primitives.
"""
def decarbonization_yield(gdp,emission_change):
    return round(max(0,1-(emission_change/max(gdp,0.1))),3)

def resilience_index(renewables,disasters):
    return round(renewables/max(disasters+1,1),3)

def carbon_liability(emission_change,offset_cost):
    return round(max(0,emission_change*offset_cost),2)

if __name__=="__main__":
    print("Decarb Yield:",decarbonization_yield(2.3,0.9))
    print("Resilience Index:",resilience_index(35,5))
    print("Carbon Liability:",carbon_liability(0.9,120))