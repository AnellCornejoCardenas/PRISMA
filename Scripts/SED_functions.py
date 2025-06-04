import numpy as np
#import astropy.constants as const
#from astropy import units as u
from pcigale import sed
from pcigale import sed_modules as modules

def bin_sample(array , bins):
    '''
    To bin an array
    Inputs  -> array: as a np.array or list
               bins: list of bins to be fixed to the sample
    Outputs -> array binned according to the "bins" list
     
    '''
    if isinstance(array, np.ndarray) == False:
        array = np.array(array)
    #idx_bin = np.digitize(array, bins)
    idx_bin_def = []
    for i in array:
        if i > max(bins): i = max(bins)  #si es mayor que el máximo lo seteamos al máximo
        else: pass
        if i in bins: idx_ = list(bins).index(i)  #si está dentro, buscamos ahí el índice
        else: idx_ = np.digitize(i,bins)          # si no, buscamos el índice con el np.digitize
        
        idx_bin_def.append(idx_)
    array_binned = bins[idx_bin_def]
    return array_binned


def stellar_sed_NLy_updated(Z, ages,SSP='cb19'):
    '''
    Inputs -> Z   : Array with stellar metallicities
              ages: Array with stellar ages (in yrs)
              SSP : String indicating the SSP model to use. Options are cb19 (Charlot and Bruzual model) or bc03 (Bruzual and Charlot model)
              
    Output -> Qion (ph/s) normalized a 1 solar mass
    
    We create this updated version of the module bc20.stellar_sed_NLy, because this module does not
    work well with arrays.
    
    This function uses the BC03 tables (extracted from the CIGALE databases) to calculate the stellar Qion.
    
    '''
                
    Qion_list = []
    for j,met in enumerate(Z):
        age = ages[j] / 1e6   # Myr
        if age<2: age = 2     # Minimum age in Myr accepted by CIGALE
            
        #met
        if SSP == 'bc03':
            Z_cigale_bins = np.array([ 0.0001,0.0004,0.004,0.008,0.02,0.05])
        elif SSP == 'cb19':
            Z_cigale_bins = np.array([ 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.014, 0.017, 0.02, 0.03, 0.04, 0.06])
        else:
            print('SSP models accepted are "cb19" or "bc03".')
            
        
        met_binned = bin_sample([met] , Z_cigale_bins)

        
        parameters = {'sfhdelayed': {'tau_main': 0.1,
                                   'age_main': age, 
                                   'tau_burst': 1.,
                                   'age_burst': 1.,
                                   'f_burst': 0.,
                                  },
                       SSP: {'imf': 1, #chabrier
                            'metallicity': met_binned, 
                            'separation_age':10,  #Myr   #separation btwn old and young pops.
                            }}
        
        cigale_sed = sed.SED()
        model = modules.get_module('sfhdelayed', **parameters['sfhdelayed'])
        model.process(cigale_sed)
        sfh = model.sfr

        # CB19 table
        model = modules.get_module(SSP,**parameters[SSP])
        model.process(cigale_sed)

        #Qion_*
        Qion_s_young = cigale_sed.info['stellar.n_ly_young']
        Qion_s_old = cigale_sed.info['stellar.n_ly_old']
        Qion_list += [cigale_sed.info['stellar.n_ly']] 
    return Qion_list




def stellar_sed_NLy_updated_BORRAR(Z, ages):
    '''
    Inputs -> Z: metallicities (array)
              ages: array with stellar ages (in yrs)
              
    Output -> Qion (ph/s) normalized a 1 solar mass
    
    We create this updated version of the module bc20.stellar_sed_NLy, because this module does not
    work well with arrays.
    
    This function uses the BC03 tables (extracted from the CIGALE databases) to calculate the stellar Qion.
    
    '''
                
    Qion_list = []
    for j,met in enumerate(Z):
        #We need to bin the age and the metallicity values to use CIGALE. Dado que CIGALE usa np.arange para definir 
        #el eje x de la SFH, necesitamos que la edad sea mayor a 2 Myr
        age = ages[j] / 1e6   #Myr
        if age<2: age = 2
            
        #met
        Z_cigale_bins = np.array([ 0.0001,0.0004,0.004,0.008,0.02,0.05])
        met_binned = bin_sample([met] , Z_cigale_bins)
        
        parameters = {'sfhdelayed': {'tau_main': 0.1,
                                   'age_main': age, 
                                   'tau_burst': 1.,
                                   'age_burst': 1.,
                                   'f_burst': 0.,
                                  },
                    'bc03': {'imf': 1, #chabrier
                            'metallicity': met_binned, 
                            'separation_age':10,  #Myr   #separation btwn old and young pops.
                            }}
        
        cigale_sed = sed.SED()
        model = modules.get_module('sfhdelayed', **parameters['sfhdelayed'])
        model.process(cigale_sed)
        sfh = model.sfr

        # BC03 table
        model = modules.get_module('bc03',**parameters['bc03'])
        model.process(cigale_sed)

        #Qion_*
        Qion_s_young = cigale_sed.info['stellar.n_ly_young']
        Qion_s_old = cigale_sed.info['stellar.n_ly_old']
        if Qion_s_young <Qion_s_old:
            print("Qyoung < Qold")
        #Qion_list.append(cigale_sed.info['stellar.n_ly'])
        Qion_list += [cigale_sed.info['stellar.n_ly']]
    return Qion_list   #CIGALE provides the normalized Qion


def stellar_sed_NLy_updated_BC20_BORRAR(Z, age, mass ,bc20_class , mass_norm=False):
    '''
    
    We create this updated version of the module bc20.stellar_sed_NLy, because this module does not
    work well with arrays
    
    '''
    spec_list , rNLy_list = [],[] 
    for j,met in enumerate(Z):
        spec, rNLy = bc20_class.stellar_sed_NLy(met, age[j], mass[j] ,mass_norm = mass_norm)
        spec_list += [spec]
        rNLy_list += [rNLy]
    return np.array(spec_list) , np.array(rNLy_list)