import sys
sys.path.append('Scripts/')  # Folder containing PRISMA's internal scripts
import Utils as uu
import Class_hex_grid as class_hex


# Data handling and file system
import pandas as pd      # For reading input data files
import numpy as np       # For numerical operations
import os                # For directory and path handling
import shutil            # For copying files and directories


# Plotting tools
import matplotlib
import matplotlib.pyplot as plt  # For plotting galaxy maps and distributions


# Geometry and spatial queries
from shapely.geometry import Polygon     # To calculate the area of each hexagonal spaxel
from scipy.spatial import KDTree         # To find the closest gas particles to young stars


# Spectra calculation and CIGALE interface
import configobj                        # To read and edit the pcigale.ini configuration files
from astropy import constants as const  # Useful physical constants
from astropy import units as u          # Units for physical quantities
import subprocess       


import pcigale as cigale
import SED_functions as SED

def synthetic_spectra_generation(df, spax, sim_name,galaxy_id,neb_f_esc,neb_f_dust,dust_att,SSP_model='bc03'):
            """
            Generates a synthetic stellar spectrum for a given spaxel using CIGALE, based on simulation data and user-defined parameters.
        
            This function creates a CIGALE configuration file from the input DataFrame, updates the main CIGALE settings (`pcigale.ini`),
            runs CIGALE to generate the synthetic spectrum, and stores the output in a structured directory according to the input parameters.
        
            Parameters
            ----------
            df : DataFrame containing the parameters required by CIGALE for the current spaxel.
            spax : Identifier of the spaxel to be processed (int).
            sim_name : Name of the simulation (string).
            galaxy_id : Identifier of the galaxy within the simulation (string).
            neb_f_esc : Escape fraction of ionizing photons used in the nebular emission model (float between 0 and 1).
            neb_f_dust : Fraction of Lyman continuum photons absorbed by dust before contributing to nebular emission (float between 0 and 1).
            dust_att : If True, includes dust attenuation using the Calzetti (2003) law in the SED fitting.
            SSP_model : Stellar population synthesis model to be used by CIGALE. Default is 'bc03'.
            """
    
    
            # -- Preparing CIGALE configuration file --
            neb_f_esc_str = str(neb_f_esc).replace(".","")  #to remove the dot in the str
            neb_f_dust_str = str(neb_f_dust).replace(".","")  #to remove the dot in the str      
            
            if dust_att == True:
                parameter_file_name = f"config_file_spax{spax}_{sim_name}-{galaxy_id}_fesc{neb_f_esc_str}_fdust{neb_f_dust_str}_clz.txt"
            elif dust_att == False:
                parameter_file_name = f"config_file_spax{spax}_{sim_name}-{galaxy_id}_fesc{neb_f_esc_str}_fdust{neb_f_dust_str}.txt"
                
            # We save the dataframe with the information of the CIGALE config. file
            df.to_csv(parameter_file_name,index=False, sep=" ")


            # -- Editing the pcigale.ini file considering the parameters in the config file --
            cigale_config = configobj.ConfigObj(f"{os.getcwd()}/pcigale.ini",
                    write_empty_values=True,
                    indent_type='  ',
                    encoding='UTF8')
            cigale_config['parameters_file'] = parameter_file_name
            if dust_att == True:
                cigale_config['sed_modules'] =  ['sfhdelayed', f'{SSP_model}', 'nebular','dustatt_calzleit', 'redshifting']
            else:
                cigale_config['sed_modules'] =  ['sfhdelayed', f'{SSP_model}', 'nebular', 'redshifting']
            cigale_config.write()

    
            # -- Running CIGALE --
            import subprocess
            exists = os.path.exists("out")                                                       
            if exists == True:                                                                   
                shutil.rmtree("out", ignore_errors=False, onerror=None) #to remove past results  
            cig = subprocess.run(['pcigale', 'run'])
            
            # Changing the name of the output directory
            condition = f"fesc{neb_f_esc_str}_fdust{neb_f_dust_str}"
            if dust_att == True:
                general_path = f"{os.getcwd()}/out_dirs/{sim_name}/Synthetic_spectra_with_dust_att"
            elif dust_att == False:
                general_path = f"{os.getcwd()}/out_dirs/{sim_name}/Synthetic_spectra"

            out_path = f"{general_path}/out_{sim_name}-{galaxy_id}_{condition}_{spax}"                         
            
            if os.path.exists(out_path) == True: shutil.rmtree(out_path, ignore_errors=False, onerror=None)
            shutil.move("out", out_path)
            shutil.move(parameter_file_name, out_path)
    
    
def processing_spaxel_info(spax, t_stars, t_gas,sim_name,galaxy_id,
                           lattice_s, lattice_g,center_s,
                          age_threshold,D_between_parts,idx_close,
                          x_ifu,y_ifu,z_ifu,
                          rot_vels_s,
                          x_ifu_gas,y_ifu_gas,z_ifu_gas,
                          rot_vels_g,
                          neb_f_esc,neb_f_dust,redshift,
                          run_cigale,dust_att, save_intrinsic_info,SSP_model='bc03'):

    '''
    Processes the properties of a given spaxel, including its stellar and gas content. 
    The spaxel is processed only if it contains at least one young stellar population (i.e., younger than the age_threshold).

    Parameters:
    - spax: spaxel id
    - t_stars: table with properties of stellar particles
    - t_gas: table with properties of gas particles
    - sim_name,galaxy_id: galaxy simulation name and galaxy id
    - lattice_s: spatial grid for stars
    - lattice_g: spatial grid for gas
    - center_s: galaxy center position of the spaxels
    - age_threshold: maximum age (in yr) to define young stellar populations
    - D_between_parts: distance between stars and gas particles
    - idx_close: indices of gas particles close to young stars
    - x_ifu, y_ifu, z_ifu: projected coordinates of stars in the IFU frame
    - rot_vels_s: velocity of stars after being rotated
    - x_ifu_gas, y_ifu_gas, z_ifu_gas: projected coordinates of gas particles in the IFU frame
    - rot_vels_g: velocity of gas after being rotated
    - neb_f_esc: escape fraction
    - neb_f_dust: dust attenuation factor
    - redshift: redshift of the galaxy
    - run_cigale: boolean flag to run CIGALE
    - dust_att : boolean flag to considere dust attenuation law
    - save_intrinsic_info: boolean flag to save intrinsic stellar properties
    - SSP_model: stellar population synthesis model to use (default: 'bc03')

    Returns:
    - young_spaxel (bool): True if the spaxel contains at least one young stellar population
    - gal_spaxs_info (dict): dictionary containing properties of the stars and gas in the spaxel
    '''
    
    part_s_idx = lattice_s[spax]  #we identify the stellar particles into the spaxel
    part_g_idx = lattice_g[spax]  #we identify the gas particles into the spaxel

    age_s_spax = np.array(t_stars["age (yr)"])[part_s_idx]
    old_idx = age_s_spax > age_threshold
    young_idx = (age_s_spax <= age_threshold)

    if sum(young_idx) > 0 : 
        young_spaxel = True

        # Old stellar particles information
        x_s_spax_old = x_ifu[part_s_idx][old_idx]
        y_s_spax_old = y_ifu[part_s_idx][old_idx]
        z_s_spax_old = z_ifu[part_s_idx][old_idx]
        vx_s_spax_old = rot_vels_s[0][part_s_idx][old_idx]
        vy_s_spax_old = rot_vels_s[1][part_s_idx][old_idx]
        vz_s_spax_old = rot_vels_s[2][part_s_idx][old_idx]
        age_s_spax_old = age_s_spax[old_idx]
        ids_s_spax_old = np.array(t_stars["ID paticle"])[part_s_idx][old_idx]
        Z_s_spax_old = np.array(t_stars["Z"])[part_s_idx][old_idx]
        m_s_spax_old = np.array(t_stars["mass (M_sun)"])[part_s_idx][old_idx]
        abund_H_s_old = np.array(t_stars["abund_H (M_sun)"])[part_s_idx][old_idx]
        abund_O_s_old = np.array(t_stars["abund_O (M_sun)"])[part_s_idx][old_idx]
        r_s_spax_old = np.sqrt(np.square(x_s_spax_old)+np.square(y_s_spax_old)+np.square(z_s_spax_old))
        flag_spax_old = np.ones(len(m_s_spax_old))*-1 #if flag==-1 -> old star
        
        # Young stellar particles information
        x_s_spax_young = x_ifu[part_s_idx][young_idx]
        y_s_spax_young = y_ifu[part_s_idx][young_idx]
        z_s_spax_young = z_ifu[part_s_idx][young_idx]
        vx_s_spax_young = rot_vels_s[0][part_s_idx][young_idx]
        vy_s_spax_young = rot_vels_s[1][part_s_idx][young_idx]
        vz_s_spax_young = rot_vels_s[2][part_s_idx][young_idx]
        age_s_spax_young = age_s_spax[young_idx]
        ids_s_spax_young = np.array(t_stars["ID paticle"])[part_s_idx][young_idx]
        abund_O_s_young = np.array(t_stars["abund_O (M_sun)"])[part_s_idx][young_idx]
        abund_Fe_s_young = np.array(t_stars["abund_Fe (M_sun)"])[part_s_idx][young_idx]
        abund_H_s_young = np.array(t_stars["abund_H (M_sun)"])[part_s_idx][young_idx]
        Z_s_spax_young = np.array(t_stars["Z"])[part_s_idx][young_idx]
        m_s_spax_young = np.array(t_stars["mass (M_sun)"])[part_s_idx][young_idx]
        r_s_spax_young = np.sqrt(np.square(x_s_spax_young)+np.square(y_s_spax_young)+np.square(z_s_spax_young))
        flag_spax_young = np.array(t_stars["flag"])[part_s_idx][young_idx] #if flag=0 -> young star , if flag=1 -> star forming gas

        
        # Gas particles information
        x_g_spax = x_ifu_gas[part_g_idx]
        y_g_spax = y_ifu_gas[part_g_idx]
        z_g_spax = z_ifu_gas[part_g_idx]
        vx_g_spax = rot_vels_g[0][part_g_idx]
        vy_g_spax = rot_vels_g[1][part_g_idx]
        vz_g_spax = rot_vels_g[2][part_g_idx]
        inst_sfr_g_spax = np.array(t_gas["sfr (M_sun/yr)"])[part_g_idx]
        ids_g_spax = np.array(t_gas["ID paticle"])[part_g_idx]
        Z_g_spax = np.array(t_gas["Z"])[part_g_idx]
        m_g_spax = np.array(t_gas["mass (M_sun)"])[part_g_idx]
        rho_g_spax = np.array(t_gas["gas density (M_sun/kpc**3)"])[part_g_idx]
        IntEnergy_g_spax = np.array(t_gas["internal energy (km/s)**2"])[part_g_idx]
        abund_HI_g_spax = np.array(t_gas["abun_HI"])[part_g_idx] #abun_HI (M_sun)
        ne_g_spax = np.array(t_gas["fractional electron number density"])[part_g_idx]    
        abund_H_g_spax = np.array(t_gas["abund_H (M_sun)"])[part_g_idx]

        ## Gas cloud information (to the young stellar pops.)
        D_s_g_close = D_between_parts[part_s_idx][young_idx]     # Gas-stellar population distance
        x_g_spax_close = x_ifu_gas[idx_close][part_s_idx][young_idx]
        y_g_spax_close = y_ifu_gas[idx_close][part_s_idx][young_idx]
        z_g_spax_close = z_ifu_gas[idx_close][part_s_idx][young_idx]
        vz_g_spax_close = rot_vels_g[2][idx_close][part_s_idx][young_idx]
        rho_g_spax_close = np.array(t_gas["gas density (M_sun/kpc**3)"])[idx_close][part_s_idx][young_idx]
        IntEnergy_g_spax_close =  np.array(t_gas["internal energy (km/s)**2"])[idx_close][part_s_idx][young_idx]
        m_g_spax_close = np.array(t_gas["mass (M_sun)"])[idx_close][part_s_idx][young_idx]
        abund_HI_g_spax_close = np.array(t_gas["abun_HI"])[idx_close][part_s_idx][young_idx] #abun_HI (M_sun)
        abund_H_g_spax_close = np.array(t_gas["abund_H (M_sun)"])[idx_close][part_s_idx][young_idx]
        abund_O_g_spax_close = np.array(t_gas["abund_O (M_sun)"])[idx_close][part_s_idx][young_idx]
        abund_Fe_g_spax_close = np.array(t_gas["abund_Fe (M_sun)"])[idx_close][part_s_idx][young_idx]
        Z_g_spax_close = np.array(t_gas["Z"])[idx_close][part_s_idx][young_idx]
        ne_g_spax_close = np.array(t_gas["fractional electron number density"])[idx_close][part_s_idx][young_idx]
        ids_g_spax_close = np.array(t_gas["ID paticle"])[idx_close][part_s_idx][young_idx]
        smthl_g_close = np.array(t_gas["smoothing_length"])[idx_close][part_s_idx][young_idx]

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Spaxel ID: ",spax)
        print( "There are: ",len(age_s_spax_old), " old stars and ",len(age_s_spax_young), " young stars")
        
        # -- Computing extra quantities --
        galocentric_R_mean = np.mean(np.concatenate((r_s_spax_old,r_s_spax_young)))  # Indicates the mean radius to the center of the galaxy in kpc (float)

        
        # -- To use CIGALE, we need to bin the metallicity of the gas and stars --
        # Metallicity of stars
        ## For Charlot and Bruzual SSP bins (CB19)
        if SSP_model == 'cb19':  #private comunication only
            Zs_cigale_bins = np.array([ 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 
                                   0.01, 0.014, 0.017, 0.02, 0.03, 0.04, 0.06])
        ## For Bruzual and Charlot SSP bins (BC03)
        elif SSP_model == 'bc03':
            Zs_cigale_bins = np.array([ 0.0001,0.0004,0.004,0.008,0.02,0.05]) 
        
        Zs_spax_old_binned = uu.bin_sample(Z_s_spax_old , Zs_cigale_bins)
        Zs_spax_young_binned = uu.bin_sample(Z_s_spax_young , Zs_cigale_bins)

        # Metallicity of gas
        Zg_cigale_bins = np.array([0.0001, 0.0004, 0.001,0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.007,
                                   0.008, 0.009,0.011, 0.012, 0.014, 0.016, 0.019, 0.020, 0.022, 0.025,
                                   0.03,0.033, 0.037, 0.041, 0.046, 0.051])
        Zg_spax_binned = uu.bin_sample(Z_g_spax , Zg_cigale_bins) 
        Zg_close_spax_binned = uu.bin_sample(Z_g_spax_close , Zg_cigale_bins)
        

        # Rate of ionizing photons emitted by young stars
        if run_cigale == True:
            rNLy_young_particles = SED.stellar_sed_NLy_updated(Z_s_spax_young, age_s_spax_young,SSP_model)  #normalized
        else:
            rNLy_young_particles = np.ones(Z_s_spax_young.shape)*0  #[0,0,...,0]
        rNLy_young_particles = rNLy_young_particles * m_s_spax_young
        

        # Temperature of the gas cloud (K)
        temp = uu.TempFromMass(m_g_spax_close,abund_H_g_spax_close,
                                   IntEnergy_g_spax_close,ne_g_spax_close)

        
        # We'll consider the highest value between the D_s_close and the smoothing length
        d_s_close_g_kpc = []
        for i in range(len(smthl_g_close)):
            if smthl_g_close[i] >= D_s_g_close[i]:
                d_s_close_g_kpc.append(smthl_g_close[i])
            else:
                d_s_close_g_kpc.append(D_s_g_close[i])
        
        d_s_close_g_kpc = np.array(d_s_close_g_kpc)       # Distance in kpc
        d_s_close_g_cm = d_s_close_g_kpc * u.kpc.to(u.cm) # Distance in cm

        
        # If we have star forming gas particles, we need to calculate some physical properties of their gas cloud
        vol_gas_cloud = m_g_spax_close / rho_g_spax_close        # Volume 
        R_gas_cloud_kpc = np.cbrt(3/(4*np.pi) * vol_gas_cloud)   # Radius of the SF regions in kpc
        R_gas_cloud_cm = R_gas_cloud_kpc * u.kpc.to(u.cm)        # Radius of the SF regions in cm

        
        # To define the SF gas particles model
        d_s_close_g_kpc = np.where(d_s_close_g_kpc != 0,d_s_close_g_kpc,R_gas_cloud_kpc) # When d=0, d is replaced by R_gas_cloud_kpc in kpc
        d_s_close_g_cm = np.where(d_s_close_g_cm != 0,d_s_close_g_cm,R_gas_cloud_cm)     # When d=0, d is replaced by R_gas_cloud_cm in cm

        
        # To estimate gas cloud properties
        value_nH  = uu.nH(abund_H_g_spax_close, m_g_spax_close ,rho_g_spax_close) # In cm^-3
        value_nH0 = abund_HI_g_spax_close * value_nH                              # In cm^-3
        value_ne  =       ne_g_spax_close * value_nH                              # In cm^-3
    
        H_recom_rate = 2*1e-13 #cm3/s   #Case B recombination coefficient

        
        # We'll set a limit on D to avoid resolution problems.
        fix_d = 50*u.pc
        d_s_close_g_kpc = np.where(d_s_close_g_kpc > (fix_d.to(u.kpc)).value , 
                                    d_s_close_g_kpc,(fix_d.to(u.kpc)).value)  #fix_d = 50 pc
        d_s_close_g_cm = np.where(d_s_close_g_cm > (fix_d.to(u.cm)).value, 
                                    d_s_close_g_cm,(fix_d.to(u.cm)).value)  #fix_d = 50 pc

        
        # Calculation of the MINIMUM rate of ionizing photons to ionize the entire gas cloud
        rNLy_minimum = H_recom_rate * ((value_ne+value_nH0)**2) * (d_s_close_g_cm**3) * (4*np.pi/3)
        
        # If the star emits more than the gas cloud needs:
        rNLy = np.where(rNLy_young_particles <= rNLy_minimum,rNLy_young_particles,
                        rNLy_minimum) #when ph_star > ph_gas I take ph_gas (i.e., min value)
        ne_value = np.where(rNLy_young_particles <= rNLy_minimum,value_ne,
                            value_ne+value_nH0) # when ph_star > ph_gas, I considered all the gas particle ionized

        #If the star emits less than the gas particle needs:
        ne_value_copy = ne_value.copy()
        ne_value = np.where(rNLy_young_particles > rNLy_minimum,ne_value_copy,np.sqrt(rNLy/(H_recom_rate*(d_s_close_g_cm**3) * (4*np.pi/3)))) # when ph_star <= ph_gas, I considere gas is ionized according to the rNLy value

        ne_bins = np.array([10,100,1000])  #CIGALE bins
        ne_young = uu.bin_sample(ne_value , ne_bins)

        
        # LogU calculation
        logU_young = np.array([uu.LOG_U_with_energy(rNLy[i],d_s_close_g_cm[i],value_nH0[i],fesc=neb_f_esc,fdust=neb_f_dust)[0][0]
                                                    for i in range(len(rNLy))])
        logU_young_value = np.array([uu.LOG_U_with_energy(rNLy[i],d_s_close_g_cm[i],value_nH0[i], fesc=neb_f_esc,fdust=neb_f_dust)[1] 
                                                          for i in range(len(rNLy))])

        
        #Setting a minimum value for logU in -3.3
        logU_young_new = []        #bins of logU (according to CIGALE bins)
        logU_young_value_new = []  #values of logU
        for i in range(len(logU_young)):
            if logU_young_value[i] < -3.3:
                logU_young_new.append(-3.3)
                logU_young_value_new.append(-3.3)
            else:
                logU_young_new.append(logU_young[i])
                logU_young_value_new.append(logU_young_value[i])
        logU_young_new, logU_young_value_new =  np.array(logU_young_new), np.array(logU_young_value_new)


        
        # -- CIGALE configuration file --
        
        ## Age limitations
        ages = np.concatenate((np.round(age_s_spax_young/1e6,2),np.round(age_s_spax_old/1e6,2)))
        age_tot_stars = np.where(ages > 2,ages,2) #2Myr  minimum age required by CIGALE

        ## Line width limitations
        lw_young_value = np.round(np.std(np.concatenate((vz_g_spax,vz_s_spax_young))),2)   #float per spaxel
        lw_young = np.ones(Zs_spax_young_binned.shape)*lw_young_value         
        if len(lw_young)==1 and  lw_young[0]==0: #for spaxels with just one young star with a linewidth = 0 km/s
            lw_young = [10]

        ## fesc and fdust parameters
        f_esc_young = np.ones(flag_spax_young.shape) * neb_f_esc
        f_dust_young = np.ones(flag_spax_young.shape) * neb_f_dust
        f_esc_old = np.ones(flag_spax_old.shape) * 1               #i.e., old population won't contribute to nebular emission
        f_dust_old = np.ones(flag_spax_old.shape) * 0
        
        
        df = pd.DataFrame()
        df['id'] = [i for i in range(len(flag_spax_young) + len(flag_spax_old))]

        ##SFHdelayed
        df['sfhdelayed.tau_main'] = 0.1 #Myr 
        df['sfhdelayed.age_main'] =  age_tot_stars #Myr
        df['sfhdelayed.age_burst'] = 1  #mínimum accepted age
        df['sfhdelayed.tau_burst'] = 1  #vamos a probar con esto
        df['sfhdelayed.f_burst'] = 0

        #SSP
        df[f'{SSP_model}.imf'] = 1.0     # Chabrier IMF 
        df[f'{SSP_model}.metallicity'] = np.concatenate((Zs_spax_young_binned,Zs_spax_old_binned))
        df[f'{SSP_model}.separation_age'] = 10.0 #Myr

        #Nebular Emission
        df['nebular.logU'] = np.concatenate((logU_young_new,[-4]*len(flag_spax_old)))                     # We set logU=-4 for old stars
        df['nebular.zgas'] = np.concatenate((Zg_close_spax_binned,np.ones(len(Zs_spax_old_binned))*0.005))# We set Z=0.005 for old stars 
        df['nebular.ne'] = np.concatenate((ne_young, np.ones(len(flag_spax_old))*10))
        df['nebular.lines_width'] = np.concatenate((lw_young,np.ones(len(Zs_spax_old_binned))*100))       # We set lw=100km/s for old stars 
        df['nebular.f_esc'] = np.concatenate((f_esc_young,f_esc_old))
        df['nebular.f_dust'] = np.concatenate((f_dust_young,f_dust_old))

        #Dust attenuation with Calzetti
        df['dustatt_calzleit.E_BVs_young']       = 0.3 #default
        df['dustatt_calzleit.E_BVs_old_factor']  = 1.0 #default
        df['dustatt_calzleit.uv_bump_amplitude'] = 0. #default
        df['dustatt_calzleit.powerlaw_slope']    = 0. #default
        
        #redshifting
        df['redshifting.redshift'] = redshift                    

        if run_cigale == True:
            synthetic_spectra_generation(df, spax, sim_name,galaxy_id,
                           neb_f_esc,neb_f_dust,dust_att)


        
        # -- Storing the intrinsic information per spaxel 
        # Spaxel properties
        spax_id_list = np.ones(len(np.concatenate((flag_spax_young,flag_spax_old))))*spax
        R_spax_galocentric_list = np.ones(len(np.concatenate((flag_spax_young,flag_spax_old)))) * galocentric_R_mean  #kpc
            
        # Stellar properties
        abund_O_stars = np.concatenate((abund_O_s_young,abund_O_s_old))
        abund_H_stars = np.concatenate((abund_H_s_young,abund_H_s_old))
        M_stars = np.concatenate((m_s_spax_young,m_s_spax_old))
        Z_stars_value = np.concatenate((Z_s_spax_young,Z_s_spax_old))
        Z_stars_binned = np.concatenate((Zs_spax_young_binned,Zs_spax_old_binned))
        flag_list = np.concatenate((flag_spax_young,flag_spax_old))

        # Gas cloud properties
        # Since we didn't model gas clouds surrounding the older populations, we fill the tables with 
        # -9999.0 or default values. However, these values won't affect the nebular emission, 
        #since we set f_esc=1 for the old population.
        M_g_cloud_list = np.concatenate((m_g_spax_close,[-9999.0]*len(flag_spax_old)))
        abund_O_gas_cloud = np.concatenate((abund_O_g_spax_close,[-9999.0]*len(flag_spax_old))) 
        abund_H_gas_cloud = np.concatenate((abund_H_g_spax_close,[-9999.0]*len(flag_spax_old))) 
        Z_gas_cloud_value = np.concatenate((Z_g_spax_close,np.ones(len(Zs_spax_old_binned))*0.005))
        Z_gas_cloud_binned = np.concatenate((Zg_close_spax_binned,np.ones(len(Zs_spax_old_binned))*0.005))
        rho_gas_cloud = np.concatenate((rho_g_spax_close,[-9999.0]*len(flag_spax_old)))
        temp_gas_cloud = np.concatenate((temp,[-9999.0]*len(flag_spax_old)))
        R_gas_cloud = np.concatenate((d_s_close_g_kpc,[-9999.0]*len(flag_spax_old))) #kpc
        Q_gas_cloud = np.concatenate((rNLy,[-9999.0]*len(flag_spax_old))) #photoionization rate(1/s)
        logU_gas_cloud_list_value = np.concatenate((logU_young_value,[-4]*len(flag_spax_old)))
        logU_gas_cloud_list_binned = np.concatenate((logU_young,[-4]*len(flag_spax_old)))
        ne_gas_cloud_intrinsic_list = np.concatenate((value_ne,[-9999.0]*len(flag_spax_old))) #direct from simulation, before ionization of the gas cloud
        ne_gas_cloud_binned_list = np.concatenate((ne_young, np.ones(len(flag_spax_old))*10))
        ne_gas_cloud_value_list = np.concatenate((ne_value,np.ones(len(flag_spax_old))*-9999.0))  
        nH0_gas_cloud_intrinsic_list  = np.concatenate((value_nH0 ,[-9999.0]*len(flag_spax_old)))
        vz_dispersion = np.round(np.std(np.concatenate((vz_g_spax,vz_s_spax_young))),2) 
        lw_list = np.concatenate((lw_young,np.ones(len(Zs_spax_old_binned))*100))
        
        
        # Storing the properties in a dictionary
        gal_spaxs_info = {}
        gal_spaxs_info['Spax_id']= spax_id_list
        gal_spaxs_info['R_spax_galocentric']= R_spax_galocentric_list
        #stellar
        gal_spaxs_info['Age_star']=age_tot_stars
        gal_spaxs_info['O_abundance_star']=abund_O_stars #M_sun
        gal_spaxs_info['H_abundance_star']=abund_H_stars #M_sun
        gal_spaxs_info['M_stars']=M_stars #M_sun
        gal_spaxs_info['Z_stars_value']=Z_stars_value
        gal_spaxs_info['flag'] = flag_list #-1=>old star , 0=>young star , 1=>SF gas particle
        #gas cloud information
        gal_spaxs_info['M_gas_cloud']=M_g_cloud_list  #-9999.0 for old stellar pops.
        gal_spaxs_info['H_abundance_gas_cloud']=abund_H_gas_cloud #-9999.0 for old stellar pops.
        gal_spaxs_info['O_abundance_gas_cloud']=abund_O_gas_cloud #-9999.0 for old stellar pops.
        gal_spaxs_info['Z_gas_cloud_value']=Z_gas_cloud_value 
        gal_spaxs_info['Z_gas_cloud_binned']=Z_gas_cloud_binned 
        gal_spaxs_info['rho_gas_cloud']=rho_gas_cloud
        gal_spaxs_info['temp_gas_close']=temp_gas_cloud #K
        gal_spaxs_info['R_gas_cloud']=R_gas_cloud #kpc
        gal_spaxs_info['Q_gas_cloud']=Q_gas_cloud #(1/s)                    
        gal_spaxs_info['logU_value_gas_cloud']=logU_gas_cloud_list_value
        gal_spaxs_info['logU_binned_gas_cloud']=logU_gas_cloud_list_binned  #cigale
        gal_spaxs_info['ne_gas_cloud_intrinsic']=ne_gas_cloud_intrinsic_list  #from sim
        gal_spaxs_info['ne_gas_cloud_binned']=ne_gas_cloud_binned_list #cigale binned cm^-3
        gal_spaxs_info['ne_gas_cloud_value']=ne_gas_cloud_value_list #cm^-3, after ionization
        gal_spaxs_info['nH0_gas_cloud_intrinsic']=nH0_gas_cloud_intrinsic_list #cm^-3
        gal_spaxs_info['line_width_gas_cloud']=lw_list  #km/s
        
        gal_spaxs_info = pd.DataFrame(gal_spaxs_info)
            
        return young_spaxel , gal_spaxs_info

    else:
        return False , {}


def galaxy_plotting(limit,grid_s,grid_g,
                   m_ifu,m_ifu_gas,
                    sim_name, galaxy_id,cell_size,
                   young_spaxels,center_spax):

    """
    Plot the spatial distribution and mass content of stellar and gas particles in a galaxy.

    Parameters
    limit : float
        Half-width of the plot region in kpc.
    grid_s : hex_grid object
        Hexagonal grid object for stellar particles.
    grid_g : hex_grid object
        Hexagonal grid object for gas particles.
    m_ifu : array-like
        Mass of the stellar particles.
    m_ifu_gas : array-like
        Mass of the gas particles.
    sim_name : str
        Name of the simulation.
    galaxy_id : str
        ID of the galaxy.
    cell_size : float
        Size of each spaxel (in kpc).
    young_spaxels : list of int
        Indices of spaxels with young stellar populations to be highlighted.
    center_spax : array-like
        2D coordinates of the center of each spaxel.

    Returns
    -------
    None
        Displays a matplotlib plot with two subplots.
    """
    
    extn = np.array([-limit,limit,-limit,limit])
    fig, axs = plt.subplots(ncols=2, figsize=(17, 6))
    fig.subplots_adjust(hspace=0.5, left=0.15, right=0.93)

    ax = axs[0]
    hb_plot = grid_s.hexbin(cmap='bone',C=m_ifu,reduce_C_function=np.sum, ax=ax , bins='log')
    ax.axis(extn)
    ax.set_title(f"Stellar Particles in {sim_name}-{galaxy_id}")
    cb = fig.colorbar(hb_plot , ax=ax)
    ax.set_facecolor('xkcd:black')
    cb.set_label(r'$\log_{10}(M/M_\odot)$')
    ax.set_xlabel("X (Kpc)")
    ax.set_ylabel("Y (Kpc)")
    ax.annotate(f"Spax scale: {cell_size} kpc",(-25,-25),color="white",fontsize=17)

    ax = axs[1]
    hb_gas_plot = grid_g.hexbin(cmap='bone',C=m_ifu_gas,reduce_C_function=np.sum, bins='log',ax=ax)
    ax.axis(extn)
    ax.set_title(f"Gas Particles in {sim_name}-{galaxy_id}")
    cb = fig.colorbar(hb_plot , ax=ax)
    cb.set_label(r'$\log_{10}(M/M_\odot)$')
    ax.set_facecolor('xkcd:black')
    ax.set_xlabel("X (Kpc)")
    ax.set_ylabel("Y (Kpc)")  
    
    ax.annotate(f"Spax scale: {cell_size} kpc",(-25,-25),color="white",fontsize=17)

    for i in young_spaxels:
        polygon_s = hb_plot.get_paths()[0].vertices #provides the vertices of the hexagons
        center = center_spax[i]
        vertixes = center + polygon_s
        axs[0].fill(*vertixes.T, 'c-',linewidth=0.5,edgecolor="k",alpha=1)
        axs[1].fill(*vertixes.T, 'y-',linewidth=0.5,edgecolor="k",alpha=1)
    plt.show()


def PRISMA(t_gas, t_stars, 
           sim_name, galaxy_id,cell_size,angle,rot_axis,limit,redshift,
           neb_f_esc,neb_f_dust,age_threshold,
           run_cigale, save_intrinsic_info,plott,dust_att,SSP_model='bc03'):
    """
    Main function to run the PRISMA code and generate synthetic IFU-like spectra from simulated galaxies.

    Parameters
    ----------
    t_gas , t_stars : Tables with gas and stellar particle properties.
    sim_name : Name of the simulation.
    galaxy_id : Identifier of the galaxy.
    cell_size : Size (in kpc) of each spaxel (defaul is 1).
    angle : Rotation angle in degrees.
    rot_axis : Axis of rotation ('x', 'y', or 'z').
    limit : IFU field of view in kpc.
    redshift : Redshift of the galaxy.
    neb_f_esc : Escape fraction of ionizing photons (used for nebular emission).
    neb_f_dust : Fraction of ionizing photons absorbed by dust.
    age_threshold : Age threshold (in years) to consider stellar populations as "young".
    run_cigale : Whether to run CIGALE to compute synthetic spectra (bool).
    save_intrinsic_info : Whether to save the intrinsic properties of the spaxels (bool).
    plott : Whether to plot the galaxy and its spaxels (bool).
    dust_att : If True, applies internal dust attenuation (bool).
    SSP_model : Stellar population synthesis model to use (default is 'bc03').

    Returns
    -------
    gal_spax_total :DataFrame containing the intrinsic of all spaxels that have young stellar populations.
    """

    # -- Rotation of the galaxy --
    ## Stellar particles rotation
    rot_coords_s , rot_vels_s = uu.rot_function(rot_axis , angle , np.array([t_stars["x (kpc)"],t_stars["y (kpc)"],t_stars["z (kpc)"]]), np.array([t_stars["vx (km/s)"],t_stars["vy (km/s)"],t_stars["vz (km/s)"]]))
    
    ## Gas particles rotation
    rot_coords_g , rot_vels_g = uu.rot_function(rot_axis , angle , np.array([t_gas["x (kpc)"],t_gas["y (kpc)"],t_gas["z (kpc)"]]),np.array([t_gas["vx (km/s)"],t_gas["vy (km/s)"],t_gas["vz (km/s)"]]))


    
    # -- Hexagonal grid creation (represent the spaxels of the galaxy) --
    ## Stars information
    x_ifu = rot_coords_s[0]
    y_ifu = rot_coords_s[1]
    z_ifu = rot_coords_s[2]
    m_ifu = t_stars["mass (M_sun)"]
    age_ifu = t_stars["age (yr)"]

    ### Stars grid
    grid_s = class_hex.hex_grid(x_ifu, y_ifu, cellsize=cell_size, extent=np.array([-limit,limit,-limit,limit]))

    ### Stars hexbin parameters
    hb = grid_s.hexbin(C=m_ifu , reduce_C_function=np.sum , bins='log')

    ## Gas information
    x_ifu_gas = rot_coords_g[0]
    y_ifu_gas = rot_coords_g[1]
    z_ifu_gas = rot_coords_g[2]
    m_ifu_gas = t_gas["mass (M_sun)"]
    
    ### Gas grid
    grid_g = class_hex.hex_grid(x_ifu_gas, y_ifu_gas, cellsize=cell_size, extent=np.array([-limit,limit,-limit,limit]))
    
    ### Gas hexbin parameters
    hb_gas = grid_g.hexbin(C=m_ifu_gas , reduce_C_function=np.sum , bins='log')


    
    # -- Identification of the center and area of the spaxels --
    center_s = hb.get_offsets()                 # Center of each hexagon that contains stellar particles
    polygon_s = hb.get_paths()[0].vertices      # Vertices of the generic hexagon
    area_polygon_s = Polygon(polygon_s).area    # Area in kpc

    
    # -- Checking if there are repeated ID particles --
    lattice_s, n_accum_s = grid_s.grid()
    lattice_g, n_accum_g = grid_g.grid()
    
    ## The following lines print a warning message when there are repeated particles IDs
    reps= uu.check_no_repeated_particles(lattice_s , n_accum_s , t_stars["ID paticle"])
    repg= uu.check_no_repeated_particles(lattice_g , n_accum_g , t_gas["ID paticle"]) #if there are not repeated particles, then we can continue

    

    # -- Closest gas particle to calculate nebular emission --
    tree = KDTree(np.array((t_gas["x (kpc)"],t_gas["y (kpc)"],t_gas["z (kpc)"])).T)              # Gas information
    idx = tree.query(np.array((t_stars["x (kpc)"],t_stars["y (kpc)"],t_stars["z (kpc)"])).T,k=1) # Stars information and neighbors number = 1
    D_between_parts = idx[0]           # Distances between particles
    idx_close = idx[1]                 # Indices of the nearest gas neighbor to each stellar particle



    # -- Processing spaxel data and saving the information of spaxels with young stellar pops. --
    young_spaxels = []
    center_spax = {}
    gal_spax_total = pd.DataFrame()
    cont = 0
    for spax in range(len(lattice_s)):
        spax_boolean, gal_spax_info = processing_spaxel_info(spax, t_stars, t_gas, sim_name,galaxy_id,
                               lattice_s, lattice_g,center_s,
                              age_threshold,D_between_parts,idx_close,
                              x_ifu,y_ifu,z_ifu,
                              rot_vels_s,
                              x_ifu_gas,y_ifu_gas,z_ifu_gas,
                              rot_vels_g,
                              neb_f_esc,neb_f_dust,redshift,
                              run_cigale, dust_att, save_intrinsic_info,SSP_model)
        
        # To update the gal_spax_total dictionary with the information of each spaxel
        if spax_boolean == True:
            gal_spax_total = pd.concat([gal_spax_total, gal_spax_info], ignore_index=True)
            young_spaxels.append(spax)
        
        # To identify the center of each spaxel (for the plotting process)    
        if len(lattice_s[spax]) > 0:
            center_spax[spax] = center_s[cont]
            cont += 1

    
    # --Saving the information of spaxels --
    if save_intrinsic_info == True:
        neb_f_esc_str = str(neb_f_esc).replace(".","")  #to remove the dot in the str 
        neb_f_dust_str = str(neb_f_dust).replace(".","")  #to remove the dot in the str
        
        spx_path = f'spaxels_information/{sim_name}'
        if os.path.exists(spx_path) == False: os.makedirs(spx_path)
        
        if dust_att == True:
            gal_spax_total.to_csv(f"{spx_path}/intrinsic_information_spaxels_{sim_name}-{galaxy_id}_fesc{neb_f_esc_str}_fdust{neb_f_dust_str}_clz.csv",index=False)
        else:
            gal_spax_total.to_csv(f"{spx_path}/intrinsic_information_spaxels_{sim_name}-{galaxy_id}_fesc{neb_f_esc_str}_fdust{neb_f_dust_str}.csv",index=False)

    
    
    # -- Galaxy plotting with their spaxels
    if plott == True:
        galaxy_plotting(limit,grid_s,grid_g,
                   m_ifu,m_ifu_gas,
                    sim_name, galaxy_id,cell_size,
                   young_spaxels,center_spax)

    return gal_spax_total