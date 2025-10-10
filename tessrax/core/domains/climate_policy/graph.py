"""
Plots growth vs. emissions contradictions.
"""
import matplotlib.pyplot as plt
from climate_contradiction_detector import detect_conflicts
def plot_field():
    d=detect_conflicts()
    x=[x["gdp_growth"] for x in d]
    y=[x["emission_change"] for x in d]
    c=[x["contradiction_score"] for x in d]
    lab=[x["country"] for x in d]
    plt.scatter(x,y,c=c,cmap="coolwarm",s=220)
    for i,l in enumerate(lab):
        plt.text(x[i]+0.1,y[i],l,fontsize=8)
    plt.xlabel("GDP Growth (%)")
    plt.ylabel("Emission Change (%)")
    plt.title("Tessrax Climate Policy Contradiction Map")
    plt.colorbar(label="Contradiction Score")
    plt.grid(True)
    plt.show()
if __name__=="__main__":
    plot_field()