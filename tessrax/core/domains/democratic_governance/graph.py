"""
Plots representation vs. efficiency tension.
"""
import matplotlib.pyplot as plt
from governance_contradiction_detector import detect_governance_conflicts

def plot_field():
    data=detect_governance_conflicts()
    x=[d["decision_time_days"] for d in data]
    y=[d["voter_turnout"] for d in data]
    c=[d["contradiction_score"] for d in data]
    labels=[d["region"] for d in data]
    plt.scatter(x,y,c=c,cmap="coolwarm",s=220)
    for i,l in enumerate(labels):
        plt.text(x[i]+0.2,y[i],l,fontsize=8)
    plt.xlabel("Decision Time (days)")
    plt.ylabel("Voter Turnout")
    plt.title("Tessrax Democratic Governance Contradiction Map")
    plt.colorbar(label="Contradiction Score")
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    plot_field()