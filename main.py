# pgmpy (Probabilistic Graphical Models using Python)
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Create the Bayesian Network structure (nodes and edges)
model = BayesianNetwork([('Rain', 'Traffic'), ('Traffic', 'Accident')])

# Define the Conditional Probability Distributions (CPDs)
cpd_rain = TabularCPD(variable='Rain', variable_card=2, values=[[0.7], [0.3]])  # P(Rain)
cpd_traffic_given_rain = TabularCPD(variable='Traffic', variable_card=2, 
                                    values=[[0.4, 0.8], [0.6, 0.2]], 
                                    evidence=['Rain'], evidence_card=[2])  # P(Traffic | Rain)
cpd_accident_given_traffic = TabularCPD(variable='Accident', variable_card=2, 
                                        values=[[0.1, 0.9], [0.9, 0.1]], 
                                        evidence=['Traffic'], evidence_card=[2])  # P(Accident | Traffic)

# Add CPDs to the model
model.add_cpds(cpd_rain, cpd_traffic_given_rain, cpd_accident_given_traffic)

# Validate the model
model.check_model()

# Perform inference
inference = VariableElimination(model)
probability_of_accident = inference.query(variables=['Accident'], evidence={'Rain': 1, 'Traffic': 1})
print(probability_of_accident)
