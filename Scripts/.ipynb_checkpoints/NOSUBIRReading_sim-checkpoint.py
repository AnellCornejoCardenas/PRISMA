import h5py
import numpy as np
import random
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import os
import networkx
import vaex
import pandas as pd
#import plotly.express as px
#from plotly.subplots import make_subplots
#import plotly.graph_objects as go
from astropy.cosmology import FlatLambdaCDM  #to determine ages

def get_main_branch_unique_ids(subtree, node):
    """Gets the unique ids of the subhalos belonging to the main branch of the selected subhalo (node)"""
    mpb = [node, ]
    i = 0
    while True:
        succesors = list(subtree.successors(node))
        if len(succesors) == 0:
            break
        node = succesors[0] # select only the first succesor (main branch)
        mpb.append(node)       
    return mpb


def split_unique_id(unique_id):
    """Splits the ids assign to the subhalos by the merger tree code by snap number and subfind number """
    subfind_number = int(unique_id % 1e6)
    snap_number = int((unique_id - subfind_number) / 1e6)
    
    return snap_number, subfind_number


def rotador_J(x, y, z, vx, vy, vz, m):
    """
    To rotate according to the orientation of angular momentum (J)
    """
    jx=m*(y*vz-z*vy)
    jy=m*(z*vx-x*vz)
    jz=m*(x*vy-y*vx)
    
    m_total = np.sum(m)
    Jx_total = np.sum(jx)/m_total
    Jy_total = np.sum(jy)/m_total
    Jz_total = np.sum(jz)/m_total
    
    Jmod = np.sqrt(np.square(Jx_total)+np.square(Jy_total)+np.square(Jz_total))
    Jmodxy = np.sqrt(np.square(Jx_total)+np.square(Jy_total))
    Jmodxz = np.sqrt(np.square(Jx_total)+np.square(Jz_total))
    Jmodyz = np.sqrt(np.square(Jy_total)+np.square(Jz_total))
    
    # Ángulos de Euler
    sentita=Jmodxy/Jmod
    costita=Jz_total/Jmod
    senfi=Jx_total/Jmodxy
    cosfi=-Jy_total/Jmodxy
    sensi= 0
    cossi=1
     
    # Matriz de rotación
    a11=cossi*cosfi-costita*sensi*senfi
    a12=cossi*senfi+costita*cosfi*sensi
    a13=sensi*sentita
    a21=-sensi*cosfi-costita*senfi*cossi
    a22=-sensi*senfi+costita*cosfi*cossi
    a23=cossi*sentita
    a31=sentita*senfi
    a32=-sentita*cosfi
    a33=costita
     
    return Jx_total, Jy_total, Jz_total, Jmod, a11, a12, a13, a21, a22, a23, a31, a32, a33

def pos_co_to_phys_cm(cm,a,h): #co-moving to physical position (CM)
    aux = []
    for i in range(len(cm)):
        aux.append(cm[i] * a * h ** (-1))
        
    return(aux)


def vel_co_to_phys_cm(cm,vcm,a,h,adot): #co-moving to physical velocity (CM)
    # las componentes de v no van multiplicadas por sqrt(a) para el CM (ya está considerado por el subfind) 
    aux = []
    for i in range(len(vcm)):
        aux.append(vcm[i] + adot * cm[i] * h ** (-1))        
    return(aux)


def pos_co_to_phys(pos,a,h): #co-moving to physical position 
    
    xs=pos[:, 0]
    ys=pos[:, 1]
    zs=pos[:, 2]
    aux = a * h ** (-1)
    return(xs*aux,ys*aux,zs*aux)


def vel_co_to_phys(pos,vel,a,h,adot): #co-moving to physical velocity
    xs=pos[:, 0]
    ys=pos[:, 1]
    zs=pos[:, 2]
    vxs=vel[:, 0]
    vys=vel[:, 1]
    vzs=vel[:, 2]    
    
    return((vxs * np.sqrt(a)) + adot * xs * h ** (-1),(vys * np.sqrt(a)) + adot * ys * h ** (-1),(vzs * np.sqrt(a)) + adot * zs * h ** (-1))

def density_gas_co_to_phys_cm(density_g,a,h):
    """
    Units from 10^{10} h^{2} M_\odot ckpc^{-3}  to  M_\odot kpc^{-3}
    """
    aux = (10**10) * (h**2) * (a**(-3))
    return(density_g * aux)
    
def lookback_time(a):
    """
    To calculate the age of stellar particles for a given scale factor
    Inputs -> a: scale factor
    Return -> The age in Gyr
    """
    z = np.divide((1. - a) , a)
    cosmo = FlatLambdaCDM(H0 = 67.11 , Om0=0.3175 , Ob0=0.049) # acoording to the CIELO simulations cosmology
    age = cosmo.lookback_time(z).value
    return age

def Z(abundances , m):
    Elements = np.array(["He", "C", "Mg", "O", "Fe", "Si", "H", "N", "Ne", "S", "Ca", "Zi"])
    indx     = np.arange(0,13,1)
    metals   = np.where((Elements != "He") & (Elements != "H"))[0]
    H        = np.where((Elements=="H"))[0]
    
    X = abundances[:,H][:,0]/m
    Z = np.sum(abundances[:,metals] , axis = 1)/m
    return Z #, X



def recalculate_cm(x,y,z,vx,vy,vz,m,
                   cmx_subf,cmy_subf,cmz_subf,
                   vcmx_subf, vcmy_subf, vcmz_subf):
    # CENTROS DE MASA - Shrinking Sphere
    flag_cm = 1
    if flag_cm > 0:
        
        xsh    = np.concatenate((x, []), axis=None)
        ysh    = np.concatenate((y, []), axis=None)
        zsh    = np.concatenate((z, []), axis=None)
        msh    = np.concatenate((m, []), axis=None)
        
        vxsh = np.concatenate((vx, []), axis=None)
        vysh = np.concatenate((vy, []), axis=None)
        vzsh = np.concatenate((vz, []), axis=None)
            
        mtotal = np.sum(msh)
        na     = len(x)
        nlow   = 0.1*len(x)
        
        xpris   = xsh-cmx_subf
        ypris   = ysh-cmy_subf   
        zpris   = zsh-cmz_subf
        rpris   = np.sqrt(np.square(xpris)+np.square(ypris)+np.square(zpris))
        rmaxs   = np.amax(rpris)
        
        cont   = 0
        no_it  = 0
        while ((na>=nlow) & (na>=100)):
            a_aux = np.where(rpris <= (0.975*rmaxs))
            na = len(a_aux[0])
            if na > 0:
                xscut  = xsh[a_aux]
                yscut  = ysh[a_aux]
                zscut  = zsh[a_aux]

                vxscut  = vxsh[a_aux]
                vyscut  = vysh[a_aux]
                vzscut  = vzsh[a_aux]
                
                Mscut  = msh[a_aux]
                Mtotal = np.sum(Mscut)
                
                xcm0 = np.sum(xscut*Mscut)/Mtotal
                ycm0 = np.sum(yscut*Mscut)/Mtotal
                zcm0 = np.sum(zscut*Mscut)/Mtotal
                
                vxcm0 = np.sum(vxscut*Mscut)/Mtotal
                vycm0 = np.sum(vyscut*Mscut)/Mtotal
                vzcm0 = np.sum(vzscut*Mscut)/Mtotal
                
                xpris = xscut-xcm0
                ypris = yscut-ycm0
                zpris = zscut-zcm0
                rpris = np.sqrt(np.square(xpris)+np.square(ypris)+np.square(zpris))
                rmaxs = np.amax(rpris)
                
                xsh   = xscut
                ysh   = yscut
                zsh   = zscut
                vxsh  = vxscut
                vysh  = vyscut
                vzsh  = vzscut
                
                msh  = Mscut
                cont = cont+1
            else:
                print("No puedo iterar")  
                no_it = no_it+1

        x_cm = xcm0 
        y_cm = ycm0
        z_cm = zcm0

        vx_cm = vxcm0
        vy_cm = vycm0
        vz_cm = vzcm0
    else: # Si no se recalcula, se pone el del centro de masas del grupo (real que sale fatal)
        x_cm  = cmx_subf
        y_cm  = cmy_subf
        z_cm  = cmz_subf
        vx_cm = vcmx_subf
        vy_cm = vcmy_subf
        vz_cm = vcmz_subf
        
    return(x_cm,y_cm,z_cm,vx_cm, vy_cm, vz_cm)

    
def reading_sim_information(trees , simulation , galaxy , min_snap , max_snap, parts_type = "star"):
    """This function read the simulation information separating the data of stars and gas particles
    (originally: function components_separation() )
    Inputs:     -trees: tree of the simulation
                - simulation: sim as a .hdf5 file
                - galaxy: galaxy id
                - min_snap: minimum snapshot number to load the data
                - max_snap: maximum snashot number to load the data (could be the equal to min_snap number)
                - parts_type: string specifying which type of data do you want to load: "star", "gas" or "galaxy_props"
    """
    
    subtree = networkx.dfs_tree(trees, '12800{}'.format(galaxy))
    f = h5py.File(simulation, 'r')

    home = os.getcwd()
    dir_data = os.path.join(home, f'data_{galaxy}')  #las tablas se guardarán en home/usuario/data_galaxy
    exists = os.path.exists(dir_data)

    dir_data_tab = os.path.join(home, f'data_{galaxy}/components')
    exists = os.path.exists(dir_data_tab)


    for unique_id in get_main_branch_unique_ids(subtree, '12800{}'.format(galaxy)):
                snap_number, subfind_number = split_unique_id(int(unique_id))

                if snap_number >= min_snap and snap_number <= max_snap : 
                    snapshot = f['SnapNumber_{:d}'.format(snap_number)]  
                    header = f['SnapNumber_{:d}'.format(snap_number)+'/Header']

                    header = snapshot['Header']
                    print(header)
                    z = header['Redshift'][()]
                    h = header['HubbleParam'][()]
                    omega0 = header['Omega0'][()]
                    omegaL = header['OmegaLambda'][()]
                    omega = omegaL + omega0
                    omegaR = 0   
                    a = header['Time'][()]
                    
                    ### If the units of the positions are in kpc (like CIELO Simulation)
                    H0 = 0.1*h
                    ### If the units of the positions are in Mpc 
                    #    -> H0 = 100.*h (like EAGLE Simulation)
                    adot = a*H0*(
                            omegaL + omega0*(a ** (-3))+omegaR*(a**(-4))-(omega - 1)*(a**(-2)))**0.5


                    # Information of stars
                    Coordinates_s = snapshot['PartType4/Coordinates'][:]
                    ParticleIDs_s = snapshot['PartType4/ParticleIDs'][:]
                    Velocities_s = snapshot['PartType4/Velocities'][:]
                    SubFindNumber_s = snapshot['PartType4/SubFindNumber'][:]
                    #Circularity = snapshot['PartType4/Circularity'][:]  
                    #BindingEnergy = snapshot['PartType4/BindingEnergy'][:]
                    Form_time = snapshot ['PartType4/StellarFormationTime'][:]
                    Mass_s = snapshot['PartType4/Masses'][:] * 1e10 * h**(-1)
                    Abundances_s = snapshot['PartType4/Abundances'][:]* 1e10 * h**(-1)
                    offsets_s = snapshot['SubGroups/PartType4/Offsets']

                    # Information of gas particles
                    Coordinates_g = snapshot['PartType0/Coordinates'][:]
                    ParticleIDs_g = snapshot['PartType0/ParticleIDs'][:]
                    Velocities_g = snapshot['PartType0/Velocities'][:]
                    SubFindNumber_g = snapshot['PartType0/SubFindNumber'][:]
                    #Circularity_g = snapshot['PartType0/Circularity'][:]
                    #BindingEnergy_g = snapshot['PartType0/BindingEnergy'][:]
                    Electron_abundance_g = snapshot['PartType0/ElectronAbundance'][:]           #NEW AC
                    NeutralH_abundance_g = snapshot['PartType0/NeutralHydrogenAbundance'][:]    #NEW AC
                    Density_gas = snapshot['PartType0/Density'][:]                                # NEW
                    Mass_g = snapshot['PartType0/Masses'][:] * 1e10  * h**(-1)
                    Abundances_g = snapshot['PartType0/Abundances'][:]* 1e10 * h**(-1)
                    StarFormRate_g = snapshot['PartType0/StarFormationRate'][:]            #NEW
                    InternalEnergy_g = snapshot['PartType0/InternalEnergy'][:]               #NEW
                    offsets_g = snapshot['SubGroups/PartType0/Offsets']
                    

                    #Information of the subgroup
                    SubFindNumber = snapshot['SubGroups/SubFindNumber'][:]
                    GroupNumber = snapshot['SubGroups/GroupNumber'][:]
                    SubGroupNumber = snapshot['SubGroups/SubGroupNumber'][:]
                    SubGroupPos = snapshot['SubGroups/SubGroupPos'][:]
                    SubGroupVel = snapshot['SubGroups/SubGroupVel'][:]
                    SubGroupLen = snapshot['SubGroups/SubGroupLen'][:]
                    R200 = snapshot['Groups/Group_R_Crit200'][:]


                    group_number = GroupNumber[subfind_number]
                    r_vir = R200[group_number]

                    cm_pos = SubGroupPos[subfind_number] #Center of mass of subgroup       
                    cm_vel = SubGroupVel[subfind_number] #Center of velocities of subgroup from Subfind

                    # Transform from co-moving to physical units:
                    vcmx,vcmy,vcmz = vel_co_to_phys_cm(cm_pos,cm_vel,a,h,adot)
                    cmx,cmy,cmz = pos_co_to_phys_cm(cm_pos,a,h)  

                    # Information of stars belonging to the SubHalo
                    offset_s = offsets_s[subfind_number].astype('int')
                    star_pos = Coordinates_s[offset_s[0]:offset_s[1]]
                    star_vel = Velocities_s[offset_s[0]:offset_s[1]] 
                    starids = ParticleIDs_s[offset_s[0]:offset_s[1]]
                    ms = Mass_s[offset_s[0]:offset_s[1]]
                    age_s = lookback_time(Form_time[offset_s[0]:offset_s[1]])           
                    abundances_s = Abundances_s[offset_s[0]:offset_s[1]]                
                    IDs_s = ParticleIDs_s[offset_s[0]:offset_s[1]]                      
                    #ebin_s = BindingEnergy[offset_s[0]:offset_s[1]]
                    Form_time_s = Form_time[offset_s[0]:offset_s[1]]                    

                    vxs,vys,vzs = vel_co_to_phys(star_pos,star_vel,a,h,adot) #Physical Units
                    xs,ys,zs = pos_co_to_phys(star_pos,a,h) #Physical Units
                    


                    # offset gas en subfind_number
                    offset_g = offsets_g[subfind_number].astype('int')
                    gas_pos=Coordinates_g[offset_g[0]:offset_g[1]]
                    gas_vel=Velocities_g[offset_g[0]:offset_g[1]]
                    mg = Mass_g[offset_g[0]:offset_g[1]]
                    abundances_g = Abundances_g[offset_g[0]:offset_g[1]]                
                    IDs_g = ParticleIDs_g[offset_g[0]:offset_g[1]]                      
                    e_abundance_g = Electron_abundance_g[offset_g[0]:offset_g[1]]       
                    neutralH_abundance_g = NeutralH_abundance_g[offset_g[0]:offset_g[1]]
                    density_gas = Density_gas[offset_g[0]:offset_g[1]]                  
                    StarFormRate = StarFormRate_g[offset_g[0]:offset_g[1]]              
                    #ebin_g = BindingEnergy_g[offset_g[0]:offset_g[1]]
                    internalenergy_g = InternalEnergy_g[offset_g[0]:offset_g[1]]        

                    vxg,vyg,vzg = vel_co_to_phys(gas_pos,gas_vel,a,h,adot) #Physical Units
                    xg,yg,zg = pos_co_to_phys(gas_pos,a,h) #Physical Units
                    density_phys_g = density_gas_co_to_phys_cm(density_gas,a,h) #Physical Units
                    
                    # --------------------------------------------------------------------------------------
                    #                        Center of Mass . Shrinking Sphere (shrsph)
                    # --------------------------------------------------------------------------------------
                   
                
                    xcm, ycm, zcm, vxcm, vycm, vzcm = recalculate_cm(xs,
                                                                     ys,
                                                                     zs,
                                                                     vxs,
                                                                     vys,
                                                                     vzs,
                                                                     ms,
                                                                     cmx, cmy, cmz,
                                                                     vcmx, vcmy, vcmz)
                    

                    #Stars Particles: referred to the CM
                    xpristar=xs-xcm
                    ypristar=ys-ycm
                    zpristar=zs-zcm
                    rpristar = np.sqrt(np.square(xpristar)+np.square(ypristar)+np.square(zpristar)) 
                    vxpristar=vxs-vxcm
                    vypristar=vys-vycm
                    vzpristar=vzs-vzcm

                    #Gas Particles: referred to the CM
                    xprigas=xg-xcm
                    yprigas=yg-ycm
                    zprigas=zg-zcm
                    rprigas = np.sqrt(np.square(xprigas)+np.square(yprigas)+np.square(zprigas))
                    vxprigas=vxg-vxcm
                    vyprigas=vyg-vycm
                    vzprigas=vzg-vzcm

                    #---------------------------------------------------------------------------------------------
                    #                All baryons: Calculation Ropt, Rhm, Mgal(Ropt)
                    #---------------------------------------------------------------------------------------------
                    xgal = np.copy(xpristar) #np.concatenate((xpristar, xprigas), axis=None)    
                    ygal = np.copy(ypristar) #np.concatenate((ypristar, yprigas), axis=None)    
                    zgal = np.copy(zpristar) #np.concatenate((zpristar, zprigas), axis=None)    
                    #mstar
                    mgal = np.copy(ms)       #np.concatenate((ms, mg), axis=None)               

                    nbar = len(ms) #+ len(mg)                                                   
                    rgal = np.sqrt(np.square(xgal)+np.square(ygal)+np.square(zgal))

                    if np.sum(mgal) <= 0.123480*1e10: # Corte de ME
                        rcut = 50. * np.sqrt(np.sum(mgal)/1e10/h) * a    
                    else:
                        rcut = 30.  * a

                    # Ordenar las partículas según valor ascendente del radio
                    sort_index = np.argsort(rgal)
                    rgal_ord = rgal[sort_index]
                    mgal_ord = mgal[sort_index]

                    acut = np.where(rgal_ord <= rcut)
                    ncut = len(acut[0]) 
                    if ncut > 0:
                        rgalcut = rgal_ord[acut]
                        mgalcut = mgal_ord[acut]
                        mcut = np.sum(mgalcut)

                        massb50 = 0.83*mcut
                        maux = 0
                        t = 0
                        while t<ncut:
                            if maux<massb50:
                                maux = maux + mgalcut[t]
                                ropt=rgalcut[t]                                 ## Definition of optical radius

                            t = t+1

                    # Half Mass
                    if ncut > 0:
                        rgalcut = rgal_ord[acut]
                        mgalcut = mgal_ord[acut]
                        mcut = np.sum(mgalcut)

                        massb50 = 0.50*mcut
                        maux = 0
                        t = 0
                        while t<ncut:
                            if maux<massb50:
                                maux = maux + mgalcut[t]
                                rhm=rgalcut[t]

                            t = t+1

                    # Contar número de partículas que han entrado en el radio óptico
                    aropt = np.where(rgal <= ropt)    
                    narop = len(aropt[0])
                    npart_1halfropt = narop

                    # Definir masa dentro del radio óptico
                    if narop > 0:
                        mgal_ropt = np.sum(mgal[aropt])
                    else:
                        mgal_ropt = 0 

                    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXSTELLAR INFORMATION XXXXXXXXXXXX
                    xgal = np.copy(xpristar)  # Hay que hacer copia porque con igualdad se modifica     
                    ygal = np.copy(ypristar)
                    zgal = np.copy(zpristar)
                    vxgal = np.copy(vxpristar)       
                    vygal = np.copy(vypristar)
                    vzgal = np.copy(vzpristar)
                    mgal = np.copy(ms)
                    age_stars = np.copy(age_s)                             
                    abundances_stars = np.copy(abundances_s)                
                    IDs_stars = np.copy(IDs_s)                              
                    Form_time_s = np.copy(Form_time_s)                       
                    nbar = len(ms) 
                    #egal = np.copy(ebin_s)    
                    rgal = np.sqrt(np.square(xgal)+np.square(ygal)+np.square(zgal))
                    
                    ###### AC(10 JUN 2022): GAS INFORMATION  ##################
                    xgas = np.copy(xprigas)
                    ygas = np.copy(yprigas)
                    zgas = np.copy(zprigas)
                    vxgas = np.copy(vxprigas)
                    vygas = np.copy(vyprigas)
                    vzgas = np.copy(vzprigas)
                    mgas = np.copy(mg)
                    abundances_gas = np.copy(abundances_g)
                    density_g = np.copy(density_phys_g)                          
                    internalenergy_gas = np.copy(internalenergy_g)                       
                    IDs_gas = np.copy(IDs_g)
                    ne_gas = np.copy(e_abundance_g)   #electron density
                    nH_abundance = np.copy(neutralH_abundance_g) #abundance of neutral hydrogen atoms
                    rgas = np.sqrt(np.square(xgas)+np.square(ygas)+np.square(zgas))
                    StarFormRate = np.copy(StarFormRate)                       
                    


                    #XXXXXXXXXXXXXXXX ROTACIÓN  XXXXXXXXXXXXXXXX
                    aopts = np.where(rgal <= 1.5*ropt)
                    nopts = len(aopts[0]) 

                    xsopt = xgal[aopts]
                    ysopt = ygal[aopts]
                    zsopt = zgal[aopts]
                    rsopt = np.sqrt(np.square(xsopt)+np.square(ysopt)+np.square(zsopt))
                    vxsopt = vxgal[aopts]
                    vysopt = vygal[aopts]
                    vzsopt = vzgal[aopts]
                    msopt = mgal[aopts]
                    agesopt = age_stars[aopts]                               

                    (Jxs_total,Jys_total,Jzs_total, Jmod_s, a11, a12, a13, a21, a22,a23, a31, a32, a33) = rotador_J(xsopt,ysopt,zsopt,vxsopt, vysopt,vzsopt, msopt)
                    
                    
                    # AC (10 JUN 2022) Añadimos la rotación del gas #########################
                    ########################## AC: gas info
                    aoptg = np.where(rgas <= 1.5*ropt)
                    noptg = len(aoptg[0]) 

                    xgopt = xgas[aoptg]
                    ygopt = ygas[aoptg]
                    zgopt = zgas[aoptg]
                    rgopt = np.sqrt(np.square(xgopt)+np.square(ygopt)+np.square(zgopt))
                    vxgopt = vxgas[aoptg]
                    vygopt = vygas[aoptg]
                    vzgopt = vzgas[aoptg]
                    mgopt = mgas[aoptg]
                    (Jxg_total,Jyg_total,Jzg_total, Jmod_g, a11g, a12g, a13g, a21g, a22g,a23g, a31g, a32g, a33g) = rotador_J(xgopt,ygopt,zgopt,vxgopt, vygopt,vzgopt, mgopt)
                    ###########################
                   

                    # Roto r,v
                    rotxs=a11*xgal+a12*ygal+a13*zgal
                    rotys=a21*xgal+a22*ygal+a23*zgal
                    rotzs=a31*xgal+a32*ygal+a33*zgal
                    rotrs_esf = np.sqrt(np.square(rotxs)+np.square(rotys)+np.square(rotzs)) # 3D
                    rotrs = np.sqrt(np.square(rotxs)+np.square(rotys)) # 2D
                    # v
                    rotvxs=a11*vxgal+a12*vygal+a13*vzgal
                    rotvys=a21*vxgal+a22*vygal+a23*vzgal
                    rotvzs=a31*vxgal+a32*vygal+a33*vzgal
                    
                    ############################ AC (10 Jun 2022) roto GAS:
                    rotxg=a11*xgas+a12*ygas+a13*zgas
                    rotyg=a21*xgas+a22*ygas+a23*zgas
                    rotzg=a31*xgas+a32*ygas+a33*zgas
                    rotrg_esf = np.sqrt(np.square(rotxg)+np.square(rotyg)+np.square(rotzg)) # 3D
                    rotrg = np.sqrt(np.square(rotxg)+np.square(rotyg)) # 2D
                    # v
                    rotvxg=a11*vxgas+a12*vygas+a13*vzgas
                    rotvyg=a21*vxgas+a22*vygas+a23*vzgas
                    rotvzg=a31*vxgas+a32*vygas+a33*vzgas
                    ########################################################             

                    # vtan rotada
                    rotvfis=np.sqrt(np.square(rotvxs)+np.square(rotvys)) # 2D

                    ds = vaex.from_arrays(
                        x=rotxs,
                        y=rotys,
                        z=rotzs,
                    )

                    jxsrot = (rotys*rotvzs-rotzs*rotvys)
                    jysrot = (rotzs*rotvxs-rotxs*rotvzs)
                    jzsrot = (rotxs*rotvys-rotys*rotvxs)
                    jmods= np.sqrt(np.square(jxsrot)+np.square(jysrot)+np.square(jzsrot))     

                    navir = len(np.where(rotrs_esf <= r_vir)[0])
                    ord_rotrs = np.argsort(rotrs_esf)
                    #egal_fin = egal[ord_rotrs[navir-1]] 
                    #ett = egal - egal_fin #+ 1.5*np.sqrt(np.square(rotvxs)+np.square(rotvys)+np.square(rotvzs))
                    

                    # XXXXXXXXXXXXX Definition of a2ropt (2 times the optical radius) XXXXXXXXXXXXXXXXX
                    a2ropt = np.where(rotrs_esf < 2*ropt)[0]
                    na2ropt = len(a2ropt)
                    #ett = ett[a2ropt]
                    #jzsrot = jzsrot[a2ropt]  

                    r3D=rotrs_esf[a2ropt]
                    r2D=rotrs[a2ropt] 

                    mgal = mgal[a2ropt]
                    rotxs=rotxs[a2ropt]
                    rotys=rotys[a2ropt]
                    rotzs=rotzs[a2ropt]
                    rotvxs=rotvxs[a2ropt]
                    rotvys=rotvys[a2ropt]
                    rotvzs=rotvzs[a2ropt]
                    starids=starids[a2ropt]
                    age_stars = age_stars[a2ropt]                                
                    abundances_stars = abundances_stars[a2ropt]                  
                    IDs_stars = IDs_stars[a2ropt]                                
                    Form_time_stars = Form_time_s[a2ropt]                        
                    
                    
                    ###### AC (10 Jun 2022): mask with a2ropt for the gas ################
                    a2roptg = np.where(rotrg_esf < 2*ropt)[0]
                    na2roptg = len(a2roptg)  
                    r3Dg=rotrg_esf[a2roptg]
                    r2Dg=rotrg[a2roptg] 

                    mgas = mgas[a2roptg]
                    rotxg=rotxg[a2roptg]
                    rotyg=rotyg[a2roptg]
                    rotzg=rotzg[a2roptg]
                    rotvxg=rotvxg[a2roptg]
                    rotvyg=rotvyg[a2roptg]
                    rotvzg=rotvzg[a2roptg]
                    IDs_gas=IDs_gas[a2roptg]                                     
                    abundances_gas = abundances_gas[a2roptg]
                    ne_gas = ne_gas[a2roptg]
                    nH_abundance_gas = nH_abundance[a2roptg]
                    Density_g = density_g[a2roptg]                                
                    StarFormRate_gas = StarFormRate[a2roptg]                      
                    InternalEnergy_gas = internalenergy_gas[a2roptg]              
                                  
                        
                 
                    ################################################################################
                    
                    # Galaxia en el sistema rotado
                    xgal = np.copy(rotxs)
                    ygal = np.copy(rotys) 
                    zgal = np.copy(rotzs) 
                    rgal = np.sqrt(np.square(xgal)+np.square(ygal)+np.square(zgal))
                    vxgal = np.copy(rotvxs)
                    vygal = np.copy(rotvys) 
                    vzgal = np.copy(rotvzs)    
                    
                   
                    
                    if parts_type == "star" or parts_type == "STAR":
                        return xgal,ygal,zgal,vxgal,vygal,vzgal,starids,mgal,age_stars,abundances_stars,Form_time_stars
                    
                    elif parts_type == "gas" or parts_type == "GAS":
                        return rotxg,rotyg,rotzg,rotvxg,rotvyg,rotvzg,IDs_gas,mgas,abundances_gas,ne_gas,nH_abundance_gas,Density_g,StarFormRate_gas,z,InternalEnergy_gas
                    elif parts_type == "galaxy_props":
                        return ropt,rhm,z,h,a