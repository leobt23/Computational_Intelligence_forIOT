import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd



#############################################################################################################
def get_rules(which_rules, variables_used, ctrl, antecedents, clp_variation):
    rules = []
    if which_rules == 'old_rules':
        if 'memory_usage' in variables_used and 'processor_load' in variables_used:
            rules.append(ctrl.Rule(antecedents['memory_usage']['high'] | antecedents['processor_load']['high'], clp_variation['decrease']))
            rules.append(ctrl.Rule(antecedents['memory_usage']['low'] | antecedents['processor_load']['low'], clp_variation['increase']))
        if 'latency' in variables_used:
            rules.append(ctrl.Rule(antecedents['latency']['high'], clp_variation['increase']))
            rules.append(ctrl.Rule(antecedents['latency']['low'], clp_variation['decrease']))
        if 'output_bandwidth' in variables_used:
            rules.append(ctrl.Rule(antecedents['output_bandwidth']['low'], clp_variation['decrease']))
            rules.append(ctrl.Rule(antecedents['output_bandwidth']['high'], clp_variation['increase']))
   
    if which_rules == 'memory_usage+processor_load':
        # if both are high, then clp_variation is decreased
        rules.append(ctrl.Rule(antecedents['memory_usage']['high'] | antecedents['processor_load']['high'], clp_variation['decrease']))
        # if both are low, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['memory_usage']['low'] & antecedents['processor_load']['low'], clp_variation['increase']))
        # if both are medium, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'] & antecedents['processor_load']['medium'], clp_variation['increase']))
        # if one is low and the other is medium, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['memory_usage']['low'] & antecedents['processor_load']['medium'], clp_variation['increase']))
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'] & antecedents['processor_load']['low'], clp_variation['increase']))
    
    if which_rules == 'memory_usage+processor_load(3 mode clp)':
        # if both are high, then clp_variation is decreased
        rules.append(ctrl.Rule(antecedents['memory_usage']['high'] | antecedents['processor_load']['high'], clp_variation['decrease']))
        # if both are low, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['memory_usage']['low'] & antecedents['processor_load']['low'], clp_variation['increase']))
        # if both are medium, then clp_variation is maintained
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'] & antecedents['processor_load']['medium'], clp_variation['maintain']))
        # if one is low and the other is medium, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['memory_usage']['low'] & antecedents['processor_load']['medium'], clp_variation['increase']))
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'] & antecedents['processor_load']['low'], clp_variation['increase']))

    if which_rules == 'MU+PL+Band':
        # if MU or PL are high, then clp_variation is decreased
        rules.append(ctrl.Rule(antecedents['memory_usage']['high'] | antecedents['processor_load']['high'], clp_variation['decrease']))
        # if MU and PL are both either low or medium, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['memory_usage']['low'] & antecedents['processor_load']['low'], clp_variation['increase']))
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'] & antecedents['processor_load']['low'], clp_variation['increase']))
        rules.append(ctrl.Rule(antecedents['memory_usage']['low'] & antecedents['processor_load']['medium'], clp_variation['increase']))
        # if MU and PL are both medium, then bandwidth decides it
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'] & antecedents['processor_load']['medium'] & antecedents['output_bandwidth']['low'], clp_variation['decrease']))
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'] & antecedents['processor_load']['medium'] & antecedents['output_bandwidth']['high'], clp_variation['increase']))
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'] & antecedents['processor_load']['medium'] & antecedents['output_bandwidth']['medium'], clp_variation['increase']))

    if which_rules == 'PL (3 mode clp)':
        # if PL is high, then clp_variation is decreased
        rules.append(ctrl.Rule(antecedents['processor_load']['high'], clp_variation['decrease']))
        # if PL is low, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['processor_load']['low'], clp_variation['increase']))
        # if PL is medium, then clp_variation is maintained
        rules.append(ctrl.Rule(antecedents['processor_load']['medium'], clp_variation['maintain']))

    if which_rules == 'MU (3 mode clp)':
        # if MU is high, then clp_variation is decreased
        rules.append(ctrl.Rule(antecedents['memory_usage']['high'], clp_variation['decrease']))
        # if MU is low, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['memory_usage']['low'], clp_variation['increase']))
        # if MU is medium, then clp_variation is maintained
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'], clp_variation['maintain']))

    # for the ensemble method
    if which_rules == 'Bandwidth+Latency->NetworkEffort':
        # if bandwidth is low or latency is high, then network effort is high
        rules.append(ctrl.Rule(antecedents['output_bandwidth']['low'] | antecedents['latency']['high'], clp_variation['decrease']))
        # if both are medium, then network effort is medium
        rules.append(ctrl.Rule(antecedents['output_bandwidth']['medium'] & antecedents['latency']['medium'], clp_variation['maintain']))
        # if bandwidth is high and latency is low, then network effort is low
        rules.append(ctrl.Rule(antecedents['output_bandwidth']['high'] & antecedents['latency']['low'], clp_variation['increase']))
        # if bandwidth is medium and latency is low, then network effort is low
        rules.append(ctrl.Rule(antecedents['output_bandwidth']['medium'] & antecedents['latency']['low'], clp_variation['increase']))
        # if bandwidth is low and latency is medium, then network effort is low
        rules.append(ctrl.Rule(antecedents['output_bandwidth']['low'] & antecedents['latency']['medium'], clp_variation['increase']))

    # for the ensemble method
    if which_rules == 'MU+PL->computationalEffort':
        # if MU or PL are high, then computational effort is high
        rules.append(ctrl.Rule(antecedents['memory_usage']['high'] | antecedents['processor_load']['high'], clp_variation['decrease']))
        # if both are medium, then computational effort is medium
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'] & antecedents['processor_load']['medium'], clp_variation['maintain']))
        # if both are low, then computational effort is low
        rules.append(ctrl.Rule(antecedents['memory_usage']['low'] & antecedents['processor_load']['low'], clp_variation['increase']))
        # if MU is low and PL is medium, then computational effort is low
        rules.append(ctrl.Rule(antecedents['memory_usage']['low'] & antecedents['processor_load']['medium'], clp_variation['increase']))
        # if MU is medium and PL is low, then computational effort is low
        rules.append(ctrl.Rule(antecedents['memory_usage']['medium'] & antecedents['processor_load']['low'], clp_variation['increase']))
        
    # finally the ensemble method
    if which_rules == 'ensemble':
        # if computational effort is high or network effort is high, then clp_variation is decreased
        rules.append(ctrl.Rule(antecedents['computational_effort']['high'] | antecedents['network_effort']['high'], clp_variation['decrease']))
        # if computational effort is medium and network effort is medium, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['computational_effort']['medium'] & antecedents['network_effort']['medium'], clp_variation['increase']))
        # if computational effort is low and network effort is low, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['computational_effort']['low'] & antecedents['network_effort']['low'], clp_variation['increase']))
        # if computational effort is low and network effort is medium, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['computational_effort']['low'] & antecedents['network_effort']['medium'], clp_variation['increase']))
        # if computational effort is medium and network effort is low, then clp_variation is increased
        rules.append(ctrl.Rule(antecedents['computational_effort']['medium'] & antecedents['network_effort']['low'], clp_variation['increase']))
        
    


    return rules

# this function will receive the values of the variables, and return the suggestion (including the plot and numeric value of CLP variation)
def get_suggestion(which_rules, values, variables_used, CLPVariation_mode_n,disparity=0.25, plot=True):
    # Defining the ranges for each variable
    ranges = {
        'memory_usage': np.arange(0, 1, 0.01),
        'processor_load': np.arange(0, 1, 0.01),
        'output_bandwidth': np.arange(0, 1, 0.01),
        'latency': np.arange(0, 1, 0.01),
        'input_network_throughput': np.arange(0, 1, 0.01),
        'output_network_throughput': np.arange(0, 1, 0.01)
    }

    # Defining the Antecedents
    antecedents = {var: ctrl.Antecedent(ranges[var], var) for var in variables_used}

    # Defining the output variable
    clp_variation = ctrl.Consequent(np.arange(-1, 1, 0.01), 'clp_variation')

    # Defining the membership functions for the input variables
    for var in variables_used:
        antecedents[var].automf(3, names=['low', 'medium', 'high'])

    # Defining the membership functions for the output variable
    if CLPVariation_mode_n == 2:
        clp_variation['decrease'] = fuzz.trimf(clp_variation.universe, [-1  , -1, 0+disparity])
        clp_variation['increase'] = fuzz.trimf(clp_variation.universe, [0-disparity, 1 , 1  ])
    if CLPVariation_mode_n == 3:
        clp_variation['decrease'] = fuzz.trimf(clp_variation.universe, [-1  , -1, 0-disparity])
        clp_variation['maintain'] = fuzz.trimf(clp_variation.universe, [-1+np.abs(disparity), 0, 1-np.abs(disparity)]) #  [-1+2*disparity, 0, 1-2*disparity]
        clp_variation['increase'] = fuzz.trimf(clp_variation.universe, [0+disparity, 1 , 1  ])
    
    # Defining the rules
    rules = get_rules(which_rules, variables_used, ctrl, antecedents, clp_variation)
    
    # Creating the control system
    clp_ctrl = ctrl.ControlSystem(rules)

    # Creating the simulator
    clp_simulator = ctrl.ControlSystemSimulation(clp_ctrl)

    # Keeping only the values that will be used
    values_used = [values[var] for var in variables_used]

    # Updating the variables
    for i,var in enumerate(variables_used):
        clp_simulator.input[var] = values_used[i]
    
    # Executing the simulation
    clp_simulator.compute()

    # Checking the simulation result, and what it suggests
    if plot == True:
        clp_variation.view(sim=clp_simulator)
    
    # Obtaining the final suggestion
    suggestion = clp_simulator.output['clp_variation']

    return suggestion


def get_all_suggestions(which_rules, variables_used, CLPVariation_mode_n, disparity, df):

    # Initializing the list of suggestions
    suggestions = []
    # Iterating over all instances
    for instance in range(len(df)):
        # Updating the variable values for this instance
        values = {
            'memory_usage': df['MemoryUsage'][instance],
            'processor_load': df['ProcessorLoad'][instance],
            'output_bandwidth': df['OutBandwidth'][instance],
            'latency': df['Latency'][instance],
            'input_network_throughput': df['InpNetThroughput'][instance],
            'output_network_throughput': df['OutNetThroughput'][instance]
        }

        # Getting the suggestion
        suggestion = get_suggestion(which_rules, values, variables_used, CLPVariation_mode_n,disparity, plot=False)
        suggestions.append(suggestion)
    
    return suggestions

#############################################################################################################



# Now applying these functions to obtain the suggestions

def main():
    file_name = 'project_2_fuzzy/Project2_SampleData.csv'#'Lab10-Proj2_TestS.csv'

    # importing the data
    df = pd.read_csv(file_name)


    # memory_usage+processor_load suggestions
    rules = 'memory_usage+processor_load'
    variables_used = ['memory_usage', 'processor_load']
    CLPVariation_mode_n = 2 
    disparity = 0.25
    suggestions = get_all_suggestions(rules, variables_used, CLPVariation_mode_n, disparity, df)
    
    print(suggestions)

if __name__ == "__main__":
    main()






