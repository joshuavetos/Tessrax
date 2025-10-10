"""
Governance metabolism primitives.
"""

def participation_yield(turnout, population):
    return round(turnout*population,3)

def deliberation_efficiency(decision_time, policy_quality):
    return round(policy_quality/max(decision_time,1),3)

def legitimacy_index(turnout, dissent_rate):
    return round((turnout*(1-dissent_rate)),3)

if __name__=="__main__":
    print("Participation Yield:", participation_yield(0.6,100000))
    print("Deliberation Efficiency:", deliberation_efficiency(8,0.8))
    print("Legitimacy Index:", legitimacy_index(0.62,0.1))