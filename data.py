"""
python3.5
data processing tools

"""

import numpy as np
import pandas as pd
import codecs
import re



def check_buildings_variables(data_dico):
    """ Check if all the buildings have the same variables """
    
    building0 = list(data_dico.keys())[0]
    variables = list(data_dico[building0].keys())
    same_variables = True
    
    for x in data_dico:
        if list(data_dico[x].keys()) != variables:
            same_variables = False
    return same_variables


def load_input_data(file):
    ## -- Load data --
    data = codecs.open(file, 'r', encoding='utf-8')
    data = data.read()
    data = data.splitlines()
    
    ## -- Load outside temperature --
    outside_temp = []
    i = 0
    if data[i] == "outside_temperature_degreC":
        is_number = True
        i += 1
        while is_number:
            try:
                outside_temp.append(float(data[i]))
                i += 1
            except:
                is_number = False 
    data = data[i:]
    
    ## -- Load parameters of the buildings --
    data_dico = {}
    num_building = None
    building_names = []

    for x in data:
        if x[:8] == "building":  # Parameter of a new building
            num_building = x.replace(' ', '')
            data_dico[num_building] = {}
            building_names.append(num_building)
        else:
            name_var = re.findall('[^:]*:', x)[0][:-1]  # Name of the variable
            values_var = re.findall('(?<=:).*', x)[0]
            if '[' in values_var and ']' in values_var:  # The value of the variable is a list
                values_var = values_var.replace('[', '')
                values_var = values_var.replace(']', '')
                values_var = values_var.split(',')
                values_var = [float(y) for y in values_var]
            else:   # The value of the variable is juste a float
                values_var = float(values_var)
            data_dico[num_building][name_var] = values_var
            
    same_variables = check_buildings_variables(data_dico)
    if not same_variables:
        print("PROBLEM: the buildings have not the same parameters")
    
    data_dico['buildings'] = building_names

    return outside_temp, data_dico


def load_output_data(file, outside_temp, building_dict):
    """
    loads output csv file
    returns dict of time series in np.arrays
    keys are building names
    columns are organised as such:
    
    office_temperature_degreC:Q_total_heating_kW:Q_total_AC_kW:Q_total_gains_kW:Q_total_kW
    """

    output = pd.read_csv(file, sep=":")
        
    len_year = len(outside_temp)
    output_dico = {}

    for i, building in enumerate(building_dict['buildings']):
        
        output_dico[building] = output.iloc[len_year*i:len_year*(i+1)].as_matrix()
    
    output_dico['fields'] = {'office_temperature_degreC': 0,
    					     'Q_total_heating_kW': 1,
    					     'Q_total_AC_kW': 2,
    					     'Q_total_gains_kW': 3,
    					     'Q_total_kW': 4}
    
    return output_dico


def features(outside_temp, building):
    """
    returns features built from building macro parameters
    and outside temperature
    
    WARNING: 'ventilation_week_ONif1' excluded so far
    """
    
    N = len(outside_temp)
    features = None
    
    hour = 0.0
    day = 1
    
    
    for i in range(N):

        keys = {}
        column = 0
        
        feature = [outside_temp[i], day, hour]
        
        keys[column] = 'outside_temp'
        column += 1
        keys[column] = 'day'
        column += 1
        keys[column] = 'hour'
        column += 1
        
        week_end = (day in [6, 7]) 
        monday = (day == 1)
        
        time_parameters = [0]*5 # 5 parameters depend on time (on top of outside_temp)
        
        if week_end:
            
            # label AC activity
            time_parameters[0] = 0
            time_parameters[1] = 0 # ??
            
            for k, h in enumerate(building['AC_WE_hours']):
                
                if np.ceil(h) == hour:
                    
                    time_parameters[0] = 1
                    time_parameters[1] = building['AC_WE_temperatures_degreC'][k]
            
            # label heating activity
            time_parameters[2] = 0
            time_parameters[3] = 0 # ??
            
            for k, h in enumerate(building['heating_WE_hours']):
                
                if np.ceil(h) == hour:
                    
                    time_parameters[2] = 1
                    time_parameters[3] = building['heating_WE_temperatures_degreC'][k]
                    
            # label ventilation
            time_parameters[4] = 0
            
        elif monday:
            
            # label AC activity
            time_parameters[0] = 0
            time_parameters[1] = 0 # ??
            
            for k, h in enumerate(building['AC_monday_hours']):
                
                if np.ceil(h) == hour:
                    
                    time_parameters[0] = 1
                    time_parameters[1] = building['AC_monday_temperatures_degreC'][k]
            
            # label heating activity
            time_parameters[2] = 0
            time_parameters[3] = 0 # ??
                
            for k, h in enumerate(building['heating_monday_hours']):
                
                if np.ceil(h) == hour:
                    
                    time_parameters[2] = 1
                    time_parameters[3] = building['heating_monday_temperatures_degreC'][k]
                
            # label ventilation
            time_parameters[4] = 0
                    
        else:
            
            # label AC activity
            time_parameters[0] = 0
            time_parameters[1] = 0 # ??
                
            for k, h in enumerate(building['AC_week_hours']):
                
                if np.ceil(h) == hour:
                    
                    time_parameters[0] = 1
                    time_parameters[1] = building['AC_week_temperatures_degreC'][k]
                
            # label heating activity
            time_parameters[2] = 0
            time_parameters[3] = 0 # ??
            
            for k, h in enumerate(building['heating_week_hours']):
                
                if np.ceil(h) == hour:
                    
                    time_parameters[2] = 1
                    time_parameters[3] = building['heating_week_temperatures_degreC'][k]
                
            # label ventilation
            time_parameters[4] = 0
            
            for k, h in enumerate(building['ventilation_week_hours']):
                
                if np.ceil(h) == hour:
                    
                    time_parameters[4] = 1
        
        feature += time_parameters
        
        keys[column] = 'AC_on'
        column += 1
        keys[column] = 'AC_temp'
        column += 1
        keys[column] = 'heating_on'
        column += 1
        keys[column] = 'heating_temp'
        column += 1
        keys[column] = 'ventilation_on'
        column += 1
        
        
        # constant parameters
        for key in building.keys():
            
            if type(building[key]) == float:
                
                feature.append(building[key])
                
                keys[column] = key
                column += 1
            
            elif key.split('_')[0] == 'thickness':
                        
                feature += building[key]

                for k in building[key]:

                    keys[column] = key
                    column += 1
        
        # safety check
        if i == 0:
            previous_keys = keys
        
        elif keys != previous_keys:
                
            print('ERRROR: {} differs from {}'.format(keys, previous_keys))
            break

        previous_keys = keys
        
        # add feature
        if i == 0:
            features = np.zeros((N, len(keys)))
            
        features[i, :] = feature
        
        if i % 1000 == 0:
            print('{}% of feature extraction'.format(float(i/N)*100))
       
        # next hour
        hour += 1
        day += 1
        
        if hour == 24:
            hour = 0
        
        if day == 8:
            day = 1
    
    print('done')
    
    return features, keys