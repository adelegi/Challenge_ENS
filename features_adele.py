""" Extract features Adele """

import numpy as np
import pandas as pd
from tqdm import tqdm
import time

def load_useless_varr():
    """ Names of the variable that are identical for every building or identicals to an other variable"""
    return ['airchange_infiltration_m3perh', 'airchange_ventilation_m3perh',
            'AC_power_kW', 'heating_power_kW', 'surface_m2_GROU', 'surface_m2_ROOF',
            'surface_m2_INTW', 'PCs_percent_on_night_WE', 'light_percent_on_night_WE',
            'lighting_Wperm2', 'volume2capacitance_coeff', 'initial_temperature',
            'Phantom_use_kW', 'AHU_low_threshold', 'AHU_high_threshold',  
            'nb_PCs']  # nb_PCS == np_occupants


def generate_a_day(hours, values, seuil_sup=10^6, seuil_inf=0):
    """ hours must be integers! """
    output_values = []
    output_in = []
    h_prec = hours[0]
    
    for i in range(1, len(hours)):
        h_next = hours[i]
        output_values += [values[i-1] for _ in range(h_prec, h_next)]
        
        if values[i-1] < seuil_sup and values[i-1] > seuil_inf:
            output_in += [1 for _ in range(h_prec, h_next)]
        else:
            output_in += [0 for _ in range(h_prec, h_next)]
        
        h_prec = h_next
        
    return output_values, output_in

def extract_features(dico, temp, name_building, print_info=True):
    """ 
        temp: outside temperature
        dico: dico of the parameters of each building
        name_building: name of the building from which the features are extracted
    """
    n = len(temp)
    nb_jours = int(n / 24)
    nb_sem = int(nb_jours / 7)

    if print_info:
        print("{} jours et {} semaines".format(nb_jours, nb_sem))

    features = {'outside_temp': temp,
                'hour': list(range(24))*nb_jours}

    # -- Constantes --
    for var in dico[name_building]:
        if type(dico[name_building][var]) == float:
            features[var] = [dico[name_building][var] for _ in range(n)]

    # -- Non constant variables --
    for var_evol, seuil_inf, seuil_sup in [('AC_', 0, 28), ('heating_', 17.5, 100)]:
        var_evol_in = []
        var_evol_value = []
        
        for period, nb_j in [('monday_', 1), ('week_', 4)]:
            hours = [int(x) for x in dico[name_building][var_evol + period + 'hours']]
            values = dico[name_building][var_evol + period + 'temperatures_degreC']
            
            output_values, output_in = generate_a_day(hours, values, seuil_sup=seuil_sup, seuil_inf=seuil_inf)
            var_evol_value += output_values * nb_j
            var_evol_in += output_in * nb_j
        
        # WE
        var_evol_value += [var_evol_value[0] for _ in range(2*24)]
        var_evol_in += [0 for _ in range(2*24)]

        features[var_evol + 'in'] = var_evol_in * nb_sem
        features[var_evol + 'value'] = var_evol_value * nb_sem


    features_df = pd.DataFrame(features)
    return features_df

def load_all_features(dico, temp):
    start_time = time.time()
    all_features = {}

    for building in tqdm(dico['buildings']):
        all_features[building] = extract_features(dico, temp, building, print_info=False)
    print("All the features have been loaded in {} sec".format(round(time.time() - start_time, 2)))

    return all_features