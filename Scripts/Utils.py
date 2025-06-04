##Imports
import numpy as np
import astropy.constants as const
from astropy import units as u
from scipy.integrate import quad
#from pcigale import sed
#from pcigale import sed_modules as modules

#Util functions
def rotating_matrix(theta , rot_axis):
    """
    To rotate a matrix in a random orientation.
    Inputs -> theta : angle (in degrees) to rotate the matrix
              rot_axis : axis of the rotation (x , y or z)
              
    Returns -> The coefficients of the rotation matrix
    """
    cost = np.cos(theta*np.pi/180)
    sint = np.sin(theta*np.pi/180)    
    
    if rot_axis == "x" or rot_axis == "X":
        a11 , a12 , a13 = 1 , 0 , 0
        a21 , a22 , a23 = 0 , cost , sint
        a31 , a32 , a33 = 0 , -sint , cost
        
    elif rot_axis == "y" or rot_axis == "Y":
        a11 , a12 , a13 = cost , 0 , -sint
        a21 , a22 , a23 = 0 , 1 , 0
        a31 , a32 , a33 = sint , 0 , cost
        
    elif rot_axis == "z" or rot_axis == "Z":
        a11 , a12 , a13 = cost , sint , 0
        a21 , a22 , a23 = -sint , cost , 0
        a31 , a32 , a33 = 0 , 0 , 1
        
    else:
        return("The axis have to be: \"x\", \"y\" or \"z\" ")        
        
    return a11, a12, a13, a21, a22, a23, a31, a32, a33
   
    
    
def rot_function(rot_axis , angle , pos_coords , vel_coords):
    '''
    (*)
    rot_axis : "x" , "y" or "z"
    angle : between 0° and 360°
    pos_coords : [x,y,z] (as an array)
    vel_coors : [vx,vy,vz] (as an array)
    
    output -> position and velocity coordinates rotated in the given angle
    '''
    
    rot_matrix = rotating_matrix(angle , rot_axis = rot_axis)
    a11,a12,a13,a21,a22,a23,a31,a32,a33 = rot_matrix
    A = np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
    
    rot_coords = np.dot(A,pos_coords)   #x , y , z
    rot_vels = np.dot(A,vel_coords)     # v_x , v_y , v_z
    
    return rot_coords , rot_vels


def check_no_repeated_particles(lattice , n_accum , IDs):
    """
    Check if there are repeated particles in a lattice
    Inputs -> lattice
              n_acuum
              IDs
    Output -> prints a message if there are any repeated particles
    """
    idx= np.arange(n_accum.size)
    ids = []  # In ids we store the particles IDs conteined in each cell
    for i in idx:
        s = lattice[i]
        particles_ids = list(IDs[s])
        ids.append(particles_ids)
    flat_array = np.array([item for sublist in ids for item in sublist])   
    idss_unique , idss_counts = np.unique(flat_array,return_counts=True)
    if all(idss_counts == 1) == True and sum(idss_counts) == len(flat_array):
        pass
    else:
        repeated_idx = np.where(idss_counts>1)[0]
        repeated_ids = idss_unique[repeated_idx]
        print("######################################################################################")
        print(f"THERE ARE {len(repeated_ids)} REPEATED")
        print(f"Their particles IDs are: [{repeated_ids}]")

        for i in repeated_ids:
            pos = np.where(flat_array == i)[0]  #idx of repeated particles in the flat_idss array
            print(f"The ID = {i} is repeated {len(pos)}")
        return repeated_ids , pos
    
    
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

        
def bin_sample_haciaabajo(array , bins):
    '''
    (*)
    To bin an array considering the bins downwards
    '''
    if isinstance(array, np.ndarray) == False:
        array = np.array(array)
    idx_bin = np.digitize(array, bins)
    idx_bin_def = []
    for i in idx_bin:
        if i == 0:
            idx_bin_def.append(i)
        else:
            idx_bin_def.append(i-1)
    array_binned = bins[idx_bin_def]
    return array_binned


#Physical functions
def TempFromMass(mass, abund_H, ie, ne):
    """
    To calculate the temperature of the gas particles
    Inputs : mass -> total mass of the particle (M_sun)
             abund_H -> abundance of H (M_sun)
             ie -> internal energy (km/s)
             ne -> fractional electron abundance (without units)
                 
    Output : temperature in K             
    """
    GAMMA_MINUS1 = (5/3) - 1
    XH = abund_H/mass
    yHelium = (1. - XH)/(4.*XH)
    mu = (1 + 4.* yHelium)/ (1.+ yHelium + ne )
    temp = GAMMA_MINUS1 * ie * mu * 1.6726 / 1.3806 * 1e-8 # / Boltzmann constant * Proton mass
    temp = temp * 1e10 # UnitEnergy_in_cgs / UnitMass_in_g
    return temp


def nH(abundance_H,mass_gas,density_gas):
    """
    Estimation of the number density of hydrogen atoms considering that
            𝑛_𝐻 = (𝑀_𝐻 / 𝑀_𝑇) * (𝜌 / 𝑚_𝑝)
            
    Input ->  abundance_H: H abundance of the gas particle (M_sun)
              mass_gas : mass of the gas particle (M_sun)
              density_gas: density of gas particles (M_sun / Kpc^3)
            
    Output -> number density of H atoms in that particle (nH in cm^-3)
    """ 
    
    #we need to know if the function input is a number or an array
    #if type(mass_gas) == np.float32 or type(mass_gas) == np.float64:
    #    M_H = abundances_gas[6] 
    #else:
    #    M_H = abundances_gas[:,6]  #UNITS: M_sun #["He","C","Mg","O","Fe","Si","H","N","Ne","S","Ca","Zi"]        
    M_H = abundance_H   
    m_p = const.m_p.to(u.M_sun).value #UNITS: M_sun
    M_T = mass_gas                #UNITS: M_sun
    rho = density_gas * (u.M_sun / u.kpc**3).to(u.M_sun / u.cm**3) #UNITS: M_sun / cm^-3
    nH = (M_H/M_T) * (rho /m_p) #UNITS : cm^-3   number density of H atoms in the particle
    return nH 


def Tn(T,n):
    return T * 10**(-n)

def alpha_Hplus(T): #according to https://arxiv.org/pdf/astro-ph/9509107.pdf
    """
    INPUT -> T: temperature in K
    """
    T3 = Tn(T,3)
    T6 = Tn(T,6)
    return 8.4*10**(-11)  *  T**(-0.5)  *  T3**(-0.2)  *  (1+T6**(0.7))**(-1) 

def alpha_Heplus(T): #according to https://arxiv.org/pdf/astro-ph/9509107.pdf
    """
    INPUT -> T: temperature in K
    """
    return 1.5  *  10**-10  *  T**-0.6353

def alpha_d(T): #according to https://arxiv.org/pdf/astro-ph/9509107.pdf
    """
    INPUT -> T: temperature in K
    """
    return 1.9  *  10**-3  *  T**-1.5  *  np.exp(-470000.0/T)  *  (1 + 0.3*np.exp(-94000.0/T))

def alpha_Heplusplus(T): #according to https://arxiv.org/pdf/astro-ph/9509107.pdf
    """
    INPUT -> T: temperature in K
    """
    T3 = Tn(T,3)
    T6 = Tn(T,6)
    return 3.36  *  10**-10  *  T**(-1/2)  *  T3**(-0.2)  *  ( 1 + T6**0.7)**-1

def Gamma_eH0(T) :
    """
    INPUT -> T: temperature in K
    """
    T5 = Tn(T,5)
    return 5.85*10**(-11)  *  T**(1/2)  *  np.exp(-157809.1/T)  *  (1+T5**(1/2))**(-1)

def Gamma_eHe0(T) :
    """
    INPUT -> T: temperature in K
    """
    T5 = Tn(T,5)
    return 2.38  *  10**-11  *  T**(1/2)  *  np.exp(-285335.4/T)  *  ( 1 + T5**(1/2))**(-1)

def Gamma_eHeplus(T) :
    """
    INPUT -> T: temperature in K
    """
    T5 = Tn(T,5)
    return 5.68  *  10**-12  *  T**(1/2)  *  np.exp(-631515.0/T)  *  ( 1 + T5**(1/2))**(-1)


def cross_section(l,element="H0"):
    """
    Function to calculate the cross section of different elements. We take these relations from the 
    Osterbrock book (1989; equation 2.4 for H0 and He+ , equation 2.31 for He0) following the paper KWH 
    (https://arxiv.org/pdf/astro-ph/9509107.pdf).
    To use these equations it is important to have in consideration that:
              
              *  For H0,  l < 911.22 AA or  nu > 3.29 * 1e15 1/s
              *  For He+, l < 227.98 AA or nu > 13.15 * 1e15 1/s   (this is the most energetic process)
              *  For He0, l < 503.85 AA or nu > 5.95 * 1e15 1/s
              
    Inputs: l: lambda in Amstrongs (AA)
    Output: cross section of the selected element in cm^2
    
    """
    nu = (const.c.to(u.AA/u.s) / l).value     # 1/s
    
    A0 = 6.3 * 1e-18 #cm^2
    if element == "H0":
        nu1 = 3.29 * 1e15 #1/s   #frequency for the energy 13.6 eV
        ep = np.sqrt((nu / nu1) - 1)
        sigma = A0  *  (nu1/nu)**4  *  np.exp(4 - (4*np.arctan(ep)/ep)) / (1 - np.exp(-2*np.pi / ep))

    elif element == "He+":
        nu1 = 13.15 * 1e15 #1/s   #frequency for the energy 54.4 eV
        ep = np.sqrt((nu / nu1) - 1)
        sigma = A0  *  (nu1/nu)**4  *  np.exp(4 - (4*np.arctan(ep)/ep)) / (1 - np.exp(-2*np.pi / ep))
        
    elif element == "He0":
        nu = (nu / (const.c.to(u.cm/u.s))).value #to express it in units of 1/cm (required by the eq) 
        #nu1 = 5.948251482848576 * 1e15   #frequency for the energy 24.6 eV
        alpha_T = 7.83 * 1e-18 #cm^2
        nu_T = 1.983 * 1e5 #cm^-1
        beta = 1.66
        s = 2.05
        sigma = alpha_T * ( beta*(nu/nu_T)**-s  + (1-beta)*(nu/nu_T)**(-s-1))
        
    else:
        print("2:The element has to be H , He+ or He0")
        
    return sigma #cm^2

def F_integral(Lum,wavels,dlmbda,element):
    """
    Used in the F_NLy function (to calculate an integral)
    F_NLy re-calculate the rate of photoionizing photons that ionizes the gas particle at a distance d
    """
    wl_bins = wavels #longitudes de onda otorgados por el espectro
    wl_binned = bin_sample_haciaabajo([dlmbda], wl_bins) #esta función ajusta el input dlmbda a los bines que tenemos
    i_binned = np.where(wl_bins == wl_binned)[0]  #índice del wl_binned (es UN número dentro de una lista)
    L_lambda = Lum[i_binned] #Luminosidad para el elemento dlmbda
    
    if element == "H0":
        sigma = cross_section(wl_binned,element="H0")  #cm²
    elif element == "He0":
        sigma = cross_section(wl_binned,element="He0")  #cm²
    elif element == "He+":
        sigma = cross_section(wl_binned,element="He+")  #cm²
    
    return L_lambda * sigma   # (L/wavels[j] * sigma_j) correspondiente a cada dlmbda de la integral

def F_Nly(L,wl,dist, element):
    """
    Paper : https://arxiv.org/abs/astro-ph/9509107
    In this case, we will recalcute the rate of ionizing photons following the next equation:
                    
                                   F_lambda =                  1          \int(sigma_lambda * L *  d_lambda)
                                                    -------------------*   -------------------------------
                                                            d^2 * h                      lambda
                                                     
    where L is the luminosity, simga_H is the cross section of the H (10^-22 cm^2) , d is the distance 
    between the young stellar and the gas particle and h is the Planck constant. It should be noted that 
    L/lambda is provided by the tables of Bruzual and Chalot (2003).
    
    Inputs  ->  L  : luminosity in W/A (array)
                wl : wavelength in A (array)
                dist  : separation distance between the stellar and the gas particle (Kpc)
                
    """
    h = const.h.value #J*s
    d = dist*u.kpc.to("AA")   
    constant = 1 / (d**2 * h)
    L = L*(u.L_sun/u.AA).to(u.W/u.AA)  #W/AA
    
    if element == "H0":
        max_lambda = 911.22 #AA
        result , err =  quad(lambda lmbda : constant * F_integral(L,wl,lmbda,element) ,0,max_lambda)
        
    elif element == "He0":
        max_lambda = 503.85 #AA
        result , err =  quad(lambda lmbda : constant * F_integral(L,wl,lmbda,element) ,0,max_lambda)
        
    elif element == "He+":
        max_lambda = 227.98 #AA
        result , err =  quad(lambda lmbda : constant * F_integral(L,wl,lmbda,element) ,0,max_lambda)
    
    else:
        print("1:The element has to be H , He+ or He0")
        
    return result

def nH0_with_energy(ne_frac , n_H , HI_abundance_frac, T , rNLy_H0 , rNLy_He0 , rNLy_Heplus,ne_modified=False):
    """
    Paper : https://arxiv.org/abs/astro-ph/9509107
    Estimation of the number number density of neutral H (H0) considering a convergence loop of the electron
    density and an input of energy due to a photon flux. We'll use the equations (from 30) from the paper 
    KWH 1996:
    
                                           n_H+ = n_H - n_H0
                                           nHe+ = ---
                                           nHe++ = ---
                                           
                                           ne = n_H+ + nHe+ + 2*nHe+
                                           
                                           
                                           n_H0 =                    n_H*alpha_H+ 
                                                  ---------------------------------------------
                                                  alpha_H+  +  gamma_eH0  +  (gamma_photH0 / ne)
    
    In this case, we will recalcute the rate of ionizing photons following the next equation:
                                   
                                   F_Nly = \int( L(W/A) * sigma(cm^2) )  d lambda                                       )
                                                -----------------------------------
                                                     d^2(cm^2) * lambda * h(J/s)
                                                
    where L is the luminosity, simga_H is the cross section of the H (10^-22 cm^2) , d is the distance 
    between the young stellar and the gas particle and h is the Planck constant.
    
    
    
                                          
    Input -> ne_frac      : fractional electron number density with respect to the total H number density (without units)
             n_H          : number density of H (cm^-3)
             HI_abundance_frac : Fraction of neutral Hydrogen Abundance provided by sims for each gas particle (without units)
             T            : temperature of the gas particle (K)
             rNLy         : rate of ionizing photons (1/s) from GALAXEV tables
    Output -> nH0 in cm^-3 (number density of H0 atoms)
    """

    ne_old = ne_frac * n_H #we calculate the number density of electrons in cm^-3
    n_H0 = HI_abundance_frac * n_H   #we calculate the number density of H0 in cm^-3
    #nH_plus = n_H - n_H0
    for i in range(1000):
        #we apply the equations of KWH 1996
        y = 0.24 / (4 - 4*0.24)
        nH_plus = n_H - n_H0
        if i == 0:
            ne_new = nH_plus
        else:
            ne_new = nH_plus  +  nHe_plus  +  2*nHe_plusplus   #(cm^-3)
        
        n_H0 = (n_H * alpha_Hplus(T)) / (alpha_Hplus(T) + Gamma_eH0(T) + (rNLy_H0/ne_new))
        nHe_plus = y*n_H / (1 + ((alpha_Heplus(T) + alpha_d(T))/(Gamma_eHe0(T) + rNLy_He0/ne_new)) + ((Gamma_eHeplus(T) + rNLy_Heplus/ne_new)/(alpha_Heplusplus(T))))
        nHe_plusplus = (nHe_plus*(Gamma_eHeplus(T) + rNLy_Heplus/ne_new)) / (alpha_Heplusplus(T))
        #print("nH+:",nH_plus,"nHe+:",nHe_plus , "nHe++:",nHe_plusplus)
        ne_new = 0.5 * (ne_new + ne_old)
        if np.abs(ne_new - ne_old) < 1e-4:   #convergence limit
            break
        else:
            #n_H0 = (n_H * alpha_Hplus(T)) / (alpha_Hplus(T) + Gamma_eH0(T) + (rNLy_H0/ne_new))
            ne_old = ne_new

            #Then, to use CIGALE we need to bin the nH0 to standard values
    new_nH0, new_ne_new = n_H0.copy() , ne_new.copy()
    if ne_modified == True: new_ne_new = new_ne_new*1e2
    bins = np.array([10,100,1000])   #cm^-3
    new_nH0 = bin_sample([new_nH0] , bins)
    new_ne_new = bin_sample([new_ne_new] , bins)
    
    return new_nH0 , n_H0 , new_ne_new , ne_new #binned according to CIGALE  , the real value in cm^⁻3


def LOG_U_with_energy(rNLy , R , nH0,fesc=0.0,fdust=0.0):
    """
    We calculate the photoionizing rate at a distance R as:
    
                                     U = Q_ion/ (4π * R^2 *nH0 *c)
    
    Q_ion : the number of hydrogen ionizing photons emitted per second
    R : the distance from the ionization source to the inner surface of the ionized gas cloud (in cm)
    nH0 : density of nH0 of the gas particle (cm^-3) (from the simulation)
    c: speed of light in cm/s
    fesc: fraction of photon escape (fesc=0.1 represent a 10%) [0,1]
    """
    if fesc > 1: fesc = 1
    if fdust > 1: fdust = 1

    c = const.c.to("cm/s").value #cm/s
    f = 1 - fesc - fdust         #fracción de escape de fotones

    log_U = np.log10((f*rNLy) / (4*np.pi*np.square(R) * nH0 * c))
    
    #to use CIGALE we need to bin U
    new_log_U = log_U.copy() 
    log_U_bins = np.array([ -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5,-2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3,-1.2, -1.1, -1.0])
    new_log_U = bin_sample([new_log_U], log_U_bins) #per stellar particle in the shell
    
    return new_log_U , log_U  # logU binned and the real logU


def LOG_U_with_energy_original_BORRAR(rNLy , R , nH0,fesc=0.0,fdust=0.0,energy=1):
    """
    We calculate the photoionizing rate at a distance R as:
    
                                     U = Q_ion/ (4π * R^2 *nH0 *c)
    
    Q_ion : the number of hydrogen ionizing photons emitted per second
    R : the distance from the ionization source to the inner surface of the ionized gas cloud (in cm)
    nH0 : density of nH0 of the gas particle (cm^-3) (from the simulation)
    c: speed of light in cm/s
    fesc: fraction of photon escape (fesc=0.1 represent a 10%) [0,1]
    energy: total energy percentage that arrive to the closest gas particle [0,1]
    
    Nuevo cambio 18/07/2024: ahora logU depende de fesc y fdust
    """
    if fesc > 1: fesc = 1
    if fdust > 1: fdust = 1
    if energy >1: energy =1
    c = const.c.to("cm/s").value #cm/s
    f = 1 - fesc - fdust   #fracción de escape de fotones
    rNLy = rNLy*energy
    log_U = np.log10((f*rNLy) / (4*np.pi*np.square(R) * nH0 * c))
    
    #to use CIGALE we need to bin U
    new_log_U = log_U.copy() #log_U.value.copy()
    log_U_bins = np.array([ -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5,-2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3,-1.2, -1.1, -1.0])
    new_log_U = bin_sample([new_log_U], log_U_bins) #per stellar particle in the shell
    
    return new_log_U , log_U  # logU binned and the real logU


def E_BV_function(NH_total , Z_median):
    """
    Function to correlelate the nH and metallicity with the color excess (still working on that)
    """
    Z_MW = 0.0134
    Z_LMC = 0.008
    Z_SMC = 0.004
    if Z_median <= Z_SMC:
        E_BV = 10**-22.36  *  NH_total
    elif (Z_SMC < Z_median) and (Z_median < Z_MW):
        E_BV = 10**-22.19  *  NH_total
    else: #MW case
        E_BV = (1/ 6.448)  *  10**-21  *  NH_total
    return E_BV

def att_delta_BumpAmpl(Z_median):
    """
    Function to correlelate the metallicity with dust attenuation (still working on that)
    """
    Z_MW = 0.0134
    Z_LMC = 0.008
    Z_SMC = 0.004
    if Z_median <= Z_SMC:
        delta , B =  -0.5 , 0.0 #amplitude = 0
        return delta , B
    
    elif (Z_SMC < Z_median) and (Z_median < Z_MW):
        delta , B = -0.3 , 1.6 #amplitude = 1.6 for the LMC case
        return delta , B
    
    else: #MW case
        delta , B = 0 , 3.0
        return delta , B
    
    
"""#SED FUNCTIONS
def stellar_sed_NLy_updated(Z, ages):
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


def stellar_sed_NLy_updated_BC20(Z, age, mass ,bc20_class , mass_norm=False):
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
"""