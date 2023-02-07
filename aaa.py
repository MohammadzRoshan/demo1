"""
Created on Mon Dec 19 12:10:28 2022

@author: mohammad roshan 
"""
# this is true code from 20 Dec folder --> adding more time storing 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import linalg
import time
t_longterm = time.time()

#%% Geometry, time and location discretization 
r_in =0.05 #inner radius 
r_out =50 #outer radius 
T_inf_wf = 370  #temperature of the WF
T_c_avg = 343   #T of the coolant out flow, used in thermal resistance equation 

N =800 # number of divisions --> N+1 Nodal points
growth_rate =4 # growth rate or bias factor
array = np.power(np.linspace(0,1,N+1),growth_rate) # standard array with bias from 0 to 1
delta = r_in + (r_out-r_in)*array #this shows nodes locations 
drCV = np.zeros(N+1) #Control volume lengths 
drCV[0] = (delta[1] - delta[0])  #First CV close to bd
drCV[-1] = (delta[-1] - delta[-2]) #last Cv close to bd   
for i in range(1,N,1):
    drCV[i] = 0.5*(delta[i] - delta[i-1]) + 0.5*(delta[i+1] - delta[i])

#Time durations for specific time step values               
t_1 = 360 #days ; storing data every 0.5 hrs 
t_2 = 720 #days every 1 hrs 
t_3 = 720 #days  every 2 hrs 
t_4 =  1080#days every 4 hrs 
t_5 =  1080#days every 16 hrs 
t_6 = 1440 #days every 128 hrs 
t_7 = 1800 #days every 1024 hrs 

# time steps -->  hours
t_div_1= 0.5   # in hours 
t_div_2 = 1 # in hours 
t_div_3 = 2  # in hours 
t_div_4 = 4 # in hours 
t_div_5 = 16  # in hours 
t_div_6 = 128  # in hours 
t_div_7 = 1024 # in hours 

#Number of time divisions
M1 = t_1 *24 /t_div_1 # no. of temporal nodes for first time interval length 
M2 = t_2 *24 /t_div_2  # no. of temporal nodes for second time interval length 
M3 = t_3 *24 /t_div_3  # no. of temporal nodes for second time interval length
M4 = t_4 *24 /t_div_4  # no. of temporal nodes for second time interval length
M5 = t_5 *24 /t_div_5  # no. of temporal nodes for second time interval length
M6 = t_6 *24 /t_div_6  # no. of temporal nodes for second time interval length
M7 = t_7 *24 /t_div_7  # no. of temporal nodes for second time interval length

M1 = int(M1)
M2 = int(M2)
M3 = int(M3)
M4 = int(M4)
M5 = int(M5)
M6 = int(M6)
M7 = int(M7)
M = M1 + M2 + M3 + M4 + M5 +M6+M7 # Total no. of temporal nodes 
M = int(M)

#hr to seconds conversion
dt = np.zeros(7) 
dt [0] = t_div_1 * 3600 #sec
dt[1] =  t_div_2 * 3600 
dt[2] =  t_div_3 * 3600 
dt[3] =  t_div_4 * 3600 
dt[4] =  t_div_5 * 3600 
dt[5] =  t_div_6 * 3600 
dt[6] =  t_div_7 * 3600 


#%% Rock properties, Fourier calc's
k_rock = 2 # thermal conductivity of the ROCk
rho = 2490  # density of the ROCK kg/m^3
cp = 1100 #Specific heat of the ROCK make sure of unit j/kgk
alpha = k_rock/(rho*cp)   # thermal diff,
cpw = 2330 # pentane

FOe_t1 = np.zeros(N+1) #During M1
FOe_t2 = np.zeros(N+1) #During M2
FOe_t3 = np.zeros(N+1) #During M3
FOe_t4 = np.zeros(N+1) #During M4
FOe_t5 = np.zeros(N+1) #During M5
FOe_t6 = np.zeros(N+1) #During M6
FOe_t7 = np.zeros(N+1) #During M7

v = 200 # is divisions --> nodal points are v+1 and CVs are v+1 
l_evap_end = 151 #end of the evaporator 
d_h = 1 # vertical segments in meter
v_e = 100 #Evaporation lasts over "v_e" [m] of the main pipe
l_adia = 50 #adiabatic depth starting point
dhCV = np.zeros(v+1) #Control volume lengths in axial direction

for i in range(0,v+1,1):
    dhCV[i] = d_h 

# Initial conditions, temperature distribution
nodes_u = np.zeros([v+1, M+1, N + 2])
for j in range(0,v+1,1):
    if j <61:
        nodes_u[j,0,1:] = 2.0718* j + 47.988 + 273.15
    elif 61 <= j <104:
        nodes_u[j,0,1:] = 0.2285* j + 147.99+ 273.15
    elif 104 <= j <154:
        nodes_u[j,0,1:] = 0.5597* j + 115.7+ 273.15
    elif 155 <= j <253:
        nodes_u[j,0,1:] = 2E-05* j**3 - 0.0148* j**2 + 2.8693* j + 24.197+ 273.15
    elif 254 <= j < 301:
        nodes_u[j,0,1:]  = 171 + 273.15
        
        
# Filling Fourier Matrix         
FOe_t1[0] =  alpha * dt[0] * ((drCV[0]))**(-2) # works on 0 up to drCV[0] : first 
FOe_t1[-1] = alpha * dt[0] * ((drCV[-1]))**(-2) # works on 0 up to drCV[-1] : last  
for i in range(1,N,1):  #inside nodes
    FOe_t1[i] = alpha * dt[0] * ((drCV[i]/2)+(drCV[i-1]/2))**(-2)  #100 which is the same N

  
#for k in range (M1,M2,1):
FOe_t2[0] = alpha * dt[1] * ((drCV[0]))**(-2) # caz it is the first node 
FOe_t2[-1] = alpha * dt[1] * ((drCV[-1]))**(-2) # only a func of time
for i in range(1,N,1):
    FOe_t2[i] = alpha * dt[1] * ((drCV[i]/2)+(drCV[i-1]/2))**(-2)

#for k in range (M2,M3,1):
FOe_t3[0] = alpha * dt[2] * ((drCV[0]))**(-2) # caz it is the first node 
FOe_t3[-1] = alpha * dt[2] * ((drCV[-1]))**(-2) # only a func of time
for i in range(1,N,1):
    FOe_t3[i] = alpha * dt[2] * ((drCV[i]/2)+(drCV[i-1]/2))**(-2)

#for k in range (M3,M4,1):
FOe_t4[0] = alpha * dt[3] * ((drCV[0]))**(-2) # caz it is the first node 
FOe_t4[-1] = alpha * dt[3] * ((drCV[-1]))**(-2) # only a func of time
for i in range(1,N,1):
    FOe_t4[i] = alpha * dt[3] * ((drCV[i]/2)+(drCV[i-1]/2))**(-2)

#for k in range (M4,M5,1):
FOe_t5[0] = alpha * dt[4] * ((drCV[0]))**(-2) # caz it is the first node 
FOe_t5[-1] = alpha * dt[4] * ((drCV[-1]))**(-2) # only a func of time
for i in range(1,N,1):
    FOe_t5[i] = alpha * dt[4] * ((drCV[i]/2)+(drCV[i-1]/2))**(-2)


#for k in range (M5,M6,1):
FOe_t6[0] = alpha * dt[5] * ((drCV[0]))**(-2) # caz it is the first node 
FOe_t6[-1] = alpha * dt[5] * ((drCV[-1]))**(-2) # only a func of time
for i in range(1,N,1):
    FOe_t6[i] = alpha * dt[5] * ((drCV[i]/2)+(drCV[i-1]/2))**(-2)

#for k in range (M6,M7,1):
FOe_t7[0] = alpha * dt[6] * ((drCV[0]))**(-2) # caz it is the first node 
FOe_t7[-1] = alpha * dt[6] * ((drCV[-1]))**(-2) # only a func of time
for i in range(1,N,1):
    FOe_t7[i] = alpha * dt[6] * ((drCV[i]/2)+(drCV[i-1]/2))**(-2)


#%% pipe
#Evaporator - main pipe 
le = v_e*d_h # length of the evaporator section
Pipe_thick = 0.025 #[m] main pipe
Di_e = 2*r_in  #inner dia. of main pipe
Do_e = Di_e+Pipe_thick*2 # outside dia of the pipe 
k_steel = 45 # thermal cond of the steel 
# Condenser pipe and length 
lc = 2 #length of the condenser 
r_in_c = r_in
Di_c = 2*r_in_c
Do_c = Do_e

#%% Thermal resistance model 
# Conductance resistances across the heat flow are 
R1 = 1/(2*np.pi*k_steel*le) *np.log(Do_e/Di_e) 
R4 = 1/(2*np.pi*k_steel*lc) *np.log(Do_c/Di_c) 

# Convective resistances 
#R2: due to boiling of the working fluid 
P_s = 6 # saturation pressure -> @ 100 degC -- 6 bar
P_atm = 1 #atm pressure --> bar
rho_htf_l = 140  #density of WF
k_htf_l = 0.111 #SI units
cp_htf_l = cpw #SI units
g = 10
rho_htf_v = 20 #SI units
h_htf_lv = 366000 #SI units
mu_htf_l = 0.000115 #SI units
#R5: due to heat removal by the helical coil 
d_coolant =  0.025  # 1 inch times TPCT INNER DIAMETER
D_coil = Do_e + d_coolant/2   #Outer diameter of the TPCT + half of the diameter of the coil as the center of the coil from upside 

# DETERMINE FLOW REGIME -> Flow rate  
Re_cr = 2300 * (1 + 8.6 * (d_coolant/D_coil)) # critical reynolds in helica pipes 
vol_rate_cool = 5 # ml/s check out DP2 to see pressure drop

A_coolant = np.pi*d_coolant**2/4 
rho_cool = 1200 # water glycol 
miu_cool = 0.02 #pa.s 

Pr = cpw * mu_htf_l/k_htf_l  # it depends on the temperature and phase (liquid) and pressure 
Re_cool = 4 * vol_rate_cool*A_coolant*rho_cool/(np.pi*miu_cool*d_coolant)


Dn = Re_cool * (d_coolant/D_coil)**0.5 #Dean number 
#schmidt correlation; Note: !!!!!! LIMITATION OF THE CORR. to 1.5E5
if Re_cool > 100:
    Nu_s = 3.65 + 0.08*(1+0.8*(d_coolant/D_coil)**0.9)*Re_cool**(0.5+0.2903*(d_coolant/D_coil)**0.194)*Pr**0.33333
elif Re_cool < Re_cr:
    Nu_s = 3.65 + 0.08*(1+0.8*(d_coolant/D_coil)**0.9)*Re_cool**(0.5+0.2903*(d_coolant/D_coil)**0.194)*Pr**0.33333
    if Re_cool < 22000:
        Nu_s = 0.023*(1+14.8*(1+d_coolant/D_coil)*(d_coolant/D_coil)**0.3333)*Re_cool**(0.8-0.22*(d_coolant/D_coil)**0.1)*Pr**0.33333
    elif Re_cool <150000:
        Nu_s = 0.023*(4+3.6*(1-d_coolant/D_coil)*(d_coolant/D_coil)**0.8)*Re_cool**0.8*Pr**0.3333333
        

# How to calc the length of the helical coil:
#N_COIL =   # N_COIL is the number of times the wire is wound on the helix axis
S_COIL =  d_coolant
H_COIL = lc  #condenser length 
N_COIL =  H_COIL/(S_COIL +d_coolant )
len_p = np.pi*D_coil *N_COIL

if Re_cool > Re_cr: #Turbulent flow 
    f2 = 0.316/Re_cool**0.25     
    DP2 = f2 * (len_p/ d_coolant)*(rho_cool*vol_rate_cool**2/2)
else:
    f2 = 64/Re_cool
    DP2 = f2 * (len_p/ d_coolant)*(rho_cool*vol_rate_cool**2/2) 


#Nu_s: is nusselt calculated by Schmidt proposed correlation
h_5_sch = k_steel*Nu_s/D_coil #Heat transfer coefficient 
Ao_c = lc*(np.pi * Do_c**2/4)
R5 = 1/(h_5_sch * Ao_c) #Thermal resistance @ the shell and tube condneser

Qt = np.zeros(M+1)  #to save all Qt --> [W]
Qt[0] = 10000 #initial guess
Resis = np.zeros(M)

# Matrix for Horizontal calcs 
mat_dig = np.zeros([N+1,N+1])  #matrix of constatns to solve Temperature @ each horizontal row,


#%% ## ## ## VERTICAL COMPUTATIONS ## ## ## 
nodes_v = np.zeros([v + 2, M+1])
# Below: Matrix of the initial conditions of  vertical computations 
for i in range(1,v + 2,1):
    if i < 61:
        nodes_v[i,0] = 2.0718* i + 47.988 + 273.15
    elif 61 <= i <104:
        nodes_v[i,0]= 0.2285* i + 147.99+ 273.15
    elif 104 <= i <154:
        nodes_v[i,0] = 0.5597* i + 115.7+ 273.15
    elif 154 <= i <253:
        nodes_v[i,0] = 2E-05* i**3 - 0.0148* i**2 + 2.8693* i + 24.197+ 273.15
    elif 254 <= i < 301:
        nodes_v[i,0]  = 171 + 273.15
        

DTz = np.zeros([v+1, M+1])  #Vertical temperature difference is saved here, to be applied to DT_radial
h_conv_v  = 4 # W/m**2 from ground towards atmosphere

FO_v1 = alpha*dt[0]*(dhCV[2])**(-2)  # smaller tsteps 
FO_v2 = alpha*dt[1]*(dhCV[2])**(-2)  # 
FO_v3 = alpha*dt[2]*(dhCV[2])**(-2)
FO_v4 = alpha*dt[3]*(dhCV[2])**(-2)
FO_v5 = alpha*dt[4]*(dhCV[2])**(-2)
FO_v6 = alpha*dt[5]*(dhCV[2])**(-2)
FO_v7 = alpha*dt[6]*(dhCV[2])**(-2)

q_geo = 4 #note 
mat_dig_v = np.zeros([v+1,v+1])  #matrix of constatns
T_air = np.zeros([t_1 +t_2 +t_3 + t_4 +t_5 +t_6 +t_7+1])
day = 0
for k in range(0, M, 1):  
    if k < M1:
        day = k*t_div_1/24
        T_air= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
    elif M1<= k <M2:
        day = k*t_div_2/24
        T_air= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
    elif M2<= k <M3:
         day = k*t_div_3/24
         T_air= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
    
    elif M3<= k <M4:
         day = k*t_div_4/24
         T_air= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
    elif M4<= k <M5:
         day = k*t_div_5/24
         T_air= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
    elif M5<= k <M6:
         day = k*t_div_6/24
         T_air= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
    elif M6 <= k:
         day = k*t_div_7/24
         T_air= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
      
    for j in range(0,v+1,1): # whole depth (e.g., 160 [m])      
        if k <= M1:
            for i in range(1,v,1): 
                mat_dig_v[i][i] = 1.0 + 2. * FO_v1                                #main diagonal
                mat_dig_v[i][i-1] = - FO_v1                                       #-1 subsidary diag
                mat_dig_v[i][i+1] = - FO_v1                                       #+1 subsidary diag    
       
            mat_dig_v[0][0]= (k_rock/dhCV[2] + h_conv_v)/h_conv_v                       # first line of matrix, first element
            mat_dig_v[0][1]= (-k_rock/dhCV[2])/h_conv_v                                 # first line of matrix, second element     
            
            mat_dig_v[v][v-1]= -1                                       #last line of matrix N-1th element
            mat_dig_v[v][v]=  1                                           #last line of matirx Nth element 
        
        
        if   M1 < k <= M2:  
            for i in range(1,v,1): 
                mat_dig_v[i][i] = 1.0 + 2. * FO_v2                               #main diagonal
                mat_dig_v[i][i-1] = - FO_v2                                      #-1 subsidary diag
                mat_dig_v[i][i+1] = - FO_v2                                         #+1 subsidary diag    
 
            mat_dig_v[0][0]= (k_rock/(dhCV[2]) + h_conv_v)/h_conv_v                       # first line of matrix, first element
            mat_dig_v[0][1]= (-k_rock/(dhCV[2]))/h_conv_v                                 # first line of matrix, second element     
            
            mat_dig_v[v][v-1]= -1                                         #last line of matrix N-1th element
            mat_dig_v[v][v]=  1                                           #last line of matirx Nth element 
        
        if   M2 < k <= M3:  
            for i in range(1,v,1): 
                mat_dig_v[i][i] = 1.0 + 2. * FO_v3                               #main diagonal
                mat_dig_v[i][i-1] = - FO_v3                                      #-1 subsidary diag
                mat_dig_v[i][i+1] = - FO_v3                                         #+1 subsidary diag  
                
            mat_dig_v[0][0]= (k_rock/(dhCV[2]) + h_conv_v)/h_conv_v                       # first line of matrix, first element
            mat_dig_v[0][1]= (-k_rock/(dhCV[2]))/h_conv_v                                 # first line of matrix, second element     
            
            mat_dig_v[v][v-1]= -1                                         #last line of matrix N-1th element
            mat_dig_v[v][v]=  1      
        
        if   M3 < k <= M4:  
            for i in range(1,v,1): 
                mat_dig_v[i][i] = 1.0 + 2. * FO_v4                               #main diagonal
                mat_dig_v[i][i-1] = - FO_v4                                      #-1 subsidary diag
                mat_dig_v[i][i+1] = - FO_v4                                         #+1 subsidary diag    
                       
            mat_dig_v[0][0]= (k_rock/(dhCV[2]) + h_conv_v)/h_conv_v                       # first line of matrix, first element
            mat_dig_v[0][1]= (-k_rock/(dhCV[2]))/h_conv_v                                 # first line of matrix, second element     
            
            mat_dig_v[v][v-1]= -1                                         #last line of matrix N-1th element
            mat_dig_v[v][v]=  1      
        
        if   M4 < k <= M5:  
            for i in range(1,v,1): 
                mat_dig_v[i][i] = 1.0 + 2. * FO_v5                              #main diagonal
                mat_dig_v[i][i-1] = - FO_v5                                      #-1 subsidary diag
                mat_dig_v[i][i+1] = - FO_v5                                         #+1 subsidary diag    
    
            mat_dig_v[0][0]= (k_rock/(dhCV[2]) + h_conv_v)/h_conv_v                       # first line of matrix, first element
            mat_dig_v[0][1]= (-k_rock/(dhCV[2]))/h_conv_v                                 # first line of matrix, second element     
            
            mat_dig_v[v][v-1]= -1                                         #last line of matrix N-1th element
            mat_dig_v[v][v]=  1   
            
            
        if   M5 < k <= M6:  
            for i in range(1,v,1): 
                mat_dig_v[i][i] = 1.0 + 2. * FO_v6                             #main diagonal
                mat_dig_v[i][i-1] = - FO_v6                                      #-1 subsidary diag
                mat_dig_v[i][i+1] = - FO_v6                                         #+1 subsidary diag    
            mat_dig_v[0][0]= (k_rock/(dhCV[2]) + h_conv_v)/h_conv_v                       # first line of matrix, first element
            mat_dig_v[0][1]= (-k_rock/(dhCV[2]))/h_conv_v                                 # first line of matrix, second element     
                
                
            mat_dig_v[v][v-1]= -1                                         #last line of matrix N-1th element
            mat_dig_v[v][v]=  1      
    
        
        if   M6 < k :  
            for i in range(1,v,1): 
                mat_dig_v[i][i] = 1.0 + 2. * FO_v7                              #main diagonal
                mat_dig_v[i][i-1] = - FO_v7                                     #-1 subsidary diag
                mat_dig_v[i][i+1] = - FO_v7                                         #+1 subsidary diag    
            mat_dig_v[0][0]= (k_rock/(dhCV[2]) + h_conv_v)/h_conv_v                       # first line of matrix, first element
            mat_dig_v[0][1]= (-k_rock/(dhCV[2]))/h_conv_v                                 # first line of matrix, second element     

       
            mat_dig_v[v][v-1]= -1                                         #last line of matrix N-1th element
            mat_dig_v[v][v]=  1      
        rhs_v = nodes_v[1:,k].copy()                                          #Constant values @ right hand side of equation
 
        rhs_v[0] = T_air   # FLuid temp                                                       # right hand side constant for the initial boundary which is subjected to convective BCs 
        
        # print("Time step counter", "Temperature")
        # print(k, T_air)
        rhs_v[-1] = 0
        y = linalg.solve(mat_dig_v, rhs_v)   #Solving the system of algebraic eqs and computing Temperature values at all locations 
        
        nodes_v[1:, k + 1] = y.copy() 
        xx = nodes_v[1:,k+1] - nodes_v[1:,k]
        DTz[:,k+1]= xx.copy() #whole data points
        

#%% ## ## ## HORIZONTAL COMPUTATIONS ## ## ## 
T_av_along_evap = np.zeros(M+1)
T_av_along_evap[0] = np.average(nodes_u[l_adia:l_evap_end,0,1])

for k in range(0, M, 1):    
    def TotalHeat(Q):
        zi = 0.32*(rho_htf_l**0.65*k_htf_l**0.3*cp_htf_l**0.7*g**0.2*(Q/(2*np.pi*r_in*le))**0.4)/(rho_htf_v**0.25*h_htf_lv**0.4*mu_htf_l**0.1)
        h_2 = zi*(P_s/P_atm)**0.3
        h_3 = 0.925*((k_htf_l**3*rho_htf_l**2*g*h_htf_lv)/(mu_htf_l*(Q/(2*np.pi*r_in*lc))*lc))**(1/3)  #phase change
        R2 =  1/(h_2 * (2*np.pi*r_in*le))
        R3 = 1/(h_3* (2*np.pi*r_in*lc))
        Req = R1 + R2 + R3 + R4 + R5
        Resis[k] = Req.copy()
        return   Q - abs(T_c_avg - np.average(nodes_u[l_adia:l_evap_end,k,1]))/(Req) #only evaporator section  
    

        
    def derivative(f, Q,dh):
          return (f(Q+dh)-f(Q-dh)) / (2.0*dh)
      
        
    def solve(f, Q0,dh):
        lastX = Q0
        nextX = lastX + 400 *dh 
        while (abs(lastX - nextX) >dh): 
            newY = f(nextX)
            lastX = nextX
            nextX = lastX - newY / derivative(f, lastX,dh) 
        return nextX


    QFound = solve(TotalHeat, Qt[k],0.1) 
    Qt[k+1] = QFound.copy()  # Q_true is what this algorithm finds 
    
    
    
    for j in range(l_adia+1,l_evap_end+1,1):  #depths 
        if k < M1:
            for i in range (1, N, 1): # internal nodes except 0 and N
                mat_dig[i][i] = 1.0 + 2. * FOe_t1[i]   # 
                mat_dig[i][i-1] = - FOe_t1[i]*(1 - 1/(2*(r_in/drCV [i]+ i)))          #-1 diag compared with main diag,
                mat_dig[i][i+1] = - FOe_t1[i]*(1 + 1/(2*(r_in/drCV [i]+ i)))         #+1 diag compared with main diag ,
            
            # Boundaries  
            mat_dig[0][0]= 2*FOe_t1[1] +1    #heat flux bd       
            mat_dig[0][1]= -2*FOe_t1[1]     #heat flux bd  
            mat_dig[N][N-1]= -1 #constant temp
            mat_dig[N][N] = 1 #constant temp
           
        elif M1 <= k < M2:
            for i in range (1, N,1):
                mat_dig[i][i] = 1.0 + 2. * FOe_t2[i]   # is the same ,
                mat_dig[i][i-1] = - FOe_t2[i]*(1 - 1/(2*(r_in/drCV [i]+ i)))          #-1 diag compared with main diag,
                mat_dig[i][i+1] = - FOe_t2[i]*(1 + 1/(2*(r_in/drCV [i]+ i)))  
            
        #Boundary nodes     
            mat_dig[0][0]= 2*FOe_t2[1] +1   #heat flux bd        
            mat_dig[0][1]= -2*FOe_t2[1] 
            mat_dig[N][N-1]= -1 #constant temp
            mat_dig[N][N] = 1 #constant temp
            
        elif M2 <= k < M3:
            for i in range (1, N,1):
                mat_dig[i][i] = 1.0 + 2. * FOe_t3[i]   # is the same ,
                mat_dig[i][i-1] = - FOe_t3[i]*(1 - 1/(2*(r_in/drCV [i]+ i)))          #-1 diag compared with main diag,
                mat_dig[i][i+1] = - FOe_t3[i]*(1 + 1/(2*(r_in/drCV [i]+ i)))  
            
        #Boundary nodes     
            mat_dig[0][0]= 2*FOe_t3[1] +1   #heat flux bd        
            mat_dig[0][1]= -2*FOe_t3[1] 
            mat_dig[N][N-1]= -1 #constant temp
            mat_dig[N][N] = 1 #constant temp
            
        elif M3 <= k < M4:
            for i in range (1, N,1):
                mat_dig[i][i] = 1.0 + 2. * FOe_t4[i]   # is the same ,
                mat_dig[i][i-1] = - FOe_t4[i]*(1 - 1/(2*(r_in/drCV [i]+ i)))          #-1 diag compared with main diag,
                mat_dig[i][i+1] = - FOe_t4[i]*(1 + 1/(2*(r_in/drCV [i]+ i)))  
            
        #Boundary nodes     
            mat_dig[0][0]= 2*FOe_t4[1] +1   #heat flux bd        
            mat_dig[0][1]= -2*FOe_t4[1] 
            mat_dig[N][N-1]= -1 #constant temp
            mat_dig[N][N] = 1 #constant temp

        elif M4 <= k <M5:
            for i in range (1, N,1):
                mat_dig[i][i] = 1.0 + 2. * FOe_t5[i]   # is the same ,
                mat_dig[i][i-1] = - FOe_t5[i]*(1 - 1/(2*(r_in/drCV [i]+ i)))          #-1 diag compared with main diag,
                mat_dig[i][i+1] = - FOe_t5[i]*(1 + 1/(2*(r_in/drCV [i]+ i)))  
            
        #Boundary nodes     
            mat_dig[0][0]= 2*FOe_t5[1] +1   #heat flux bd        
            mat_dig[0][1]= -2*FOe_t5[1] 
            mat_dig[N][N-1]= -1 #constant temp
            mat_dig[N][N] = 1 #constant temp
            
            
        elif M5 <= k <M6:
            for i in range (1, N,1):
                mat_dig[i][i] = 1.0 + 2. * FOe_t6[i]   # is the same ,
                mat_dig[i][i-1] = - FOe_t6[i]*(1 - 1/(2*(r_in/drCV [i]+ i)))          #-1 diag compared with main diag,
                mat_dig[i][i+1] = - FOe_t6[i]*(1 + 1/(2*(r_in/drCV [i]+ i)))  
            
        #Boundary nodes     
            mat_dig[0][0]= 2*FOe_t6[1] +1   #heat flux bd        
            mat_dig[0][1]= -2*FOe_t6[1] 
            mat_dig[N][N-1]= -1 #constant temp
            mat_dig[N][N] = 1 #constant temp
            
            
        
        elif M6 <= k:
            for i in range (1, N,1):
                mat_dig[i][i] = 1.0 + 2. * FOe_t7[i]   # is the same ,
                mat_dig[i][i-1] = - FOe_t7[i]*(1 - 1/(2*(r_in/drCV [i]+ i)))          #-1 diag compared with main diag,
                mat_dig[i][i+1] = - FOe_t7[i]*(1 + 1/(2*(r_in/drCV [i]+ i)))  
            
        #Boundary nodes     
            mat_dig[0][0]= 2*FOe_t7[1] +1   #heat flux bd        
            mat_dig[0][1]= -2*FOe_t7[1] 
            mat_dig[N][N-1]= -1 #constant temp
            mat_dig[N][N] = 1 #constant temp


        nodes_u[j,k+1,0] = T_inf_wf
        
        rhs = nodes_u[j,k,1:].copy()  #previous temperautres placed as constants at RHS 
        rhs[-1] = 0
        if k < M1: 
            rhs[0] = nodes_u[j,k,1]-FOe_t1[0]*(1-delta[0]/(2*r_in))*(2*delta[0]/k_rock)*(Qt[k+1]/(2*np.pi*r_in*le)) 
        elif M1 <= k < M2: 
            rhs[0] = nodes_u[j,k,1]-FOe_t2[0]*(1-delta[0]/(2*r_in))*(2*delta[0]/k_rock)*(Qt[k+1]/(2*np.pi*r_in*le)) 
        elif M2 <= k < M3: 
            rhs[0] = nodes_u[j,k,1]-FOe_t3[0]*(1-delta[0]/(2*r_in))*(2*delta[0]/k_rock)*(Qt[k+1]/(2*np.pi*r_in*le)) 
        elif M3 <= k < M4: 
            rhs[0] = nodes_u[j,k,1]-FOe_t4[0]*(1-delta[0]/(2*r_in))*(2*delta[0]/k_rock)*(Qt[k+1]/(2*np.pi*r_in*le)) 
        elif M4 <= k<M5: 
            rhs[0] = nodes_u[j,k,1]-FOe_t5[0]*(1-delta[0]/(2*r_in))*(2*delta[0]/k_rock)*(Qt[k+1]/(2*np.pi*r_in*le)) 
        elif M5 <= k<M6: 
            rhs[0] = nodes_u[j,k,1]-FOe_t6[0]*(1-delta[0]/(2*r_in))*(2*delta[0]/k_rock)*(Qt[k+1]/(2*np.pi*r_in*le)) 
        elif M6 <= k: 
            rhs[0] = nodes_u[j,k,1]-FOe_t7[0]*(1-delta[0]/(2*r_in))*(2*delta[0]/k_rock)*(Qt[k+1]/(2*np.pi*r_in*le)) 


        x = linalg.solve(mat_dig, rhs)  #solves temperature @ one horizontal row
        x = x + DTz[j,k]
        nodes_u[j, k + 1,1:] = x.copy()
        av_temp_out_evap = np.average(nodes_u[l_adia:l_evap_end+1,k+1,1])
        T_av_along_evap[k+1] = av_temp_out_evap.copy()


#%% Post processing, Saving data (EXPORTING the results)

# 1. Temperature along the TPCT wall
np.set_printoptions(threshold=np.inf)
Ax =np.average(nodes_u[l_adia:l_evap_end,:,1])
np.savetxt("Temp_ave_along", Ax)

# 2. Temperature of selected points vs time 
# determine the depth #60, 90, 120, 150
# 60 
np.set_printoptions(threshold=np.inf)
Ax2 =nodes_u[60,M,:]
np.savetxt("Temp_60_last", Ax2)
np.set_printoptions(threshold=np.inf)
Ax3 =nodes_u[60,0,:]
np.savetxt("Temp_60_init", Ax3)

# 90
np.set_printoptions(threshold=np.inf)
Ax4 =nodes_u[90,M,:]
np.savetxt("Temp_90_last", Ax4)
np.set_printoptions(threshold=np.inf)
Ax5 =nodes_u[90,0,:]
np.savetxt("Temp_90_init", Ax5)


# 120
np.set_printoptions(threshold=np.inf)
Ax6 =nodes_u[120,M,:]
np.savetxt("Temp_120_last", Ax6)
np.set_printoptions(threshold=np.inf)
Ax7 =nodes_u[120,0,:]
np.savetxt("Temp_120_init", Ax7)


# 150
np.set_printoptions(threshold=np.inf)
Ax8 =nodes_u[150,M,:]
np.savetxt("Temp_120_last", Ax8)
np.set_printoptions(threshold=np.inf)
Ax9 =nodes_u[150,0,:]
np.savetxt("Temp_120_init", Ax9)



# 3. Q vs time 
np.set_printoptions(threshold=np.inf)
Ax1 =Qt
np.savetxt("Total_heat", Ax1)

# -------------- plots ------------------ 

# time 
# plt_x = np.zeros(M+1)
# plt_x[0] = 0
# for i in range(0,M1-1):
#     plt_x[i+1] = plt_x[i] + t_div_1/24
    
    
# plt_x[M1] = t_1
# for j in range(M1,M1+M2-1):
#     plt_x[j+1] =plt_x[j] + t_div_2/24


# plt_x[M1+M2] = t_1+t_2
# for j in range(M1+M2,M1+M2+M3-1):
#     plt_x[j+1] =plt_x[j] + t_div_3/24
    
# plt_x[M1+M2+M3] = t_1+t_2+t_3
# for j in range(M1+M2+M3,M1+M2+M3+M4-1):
#     plt_x[j+1] =plt_x[j] + t_div_4/24

# plt_x[M1+M2+M3+M4] = t_1+t_2+t_3+t_4
# for j in range(M1+M2+M3+M4,M1+M2+M3+M4+M5-1):
#     plt_x[j+1] =plt_x[j] + t_div_5/24


# plt_x[M1+M2+M3+M4+M5] = t_1+t_2+t_3+t_4+t_5
# for j in range(M1+M2+M3+M4+M5,M1+M2+M3+M4+M5+M6-1):
#     plt_x[j+1] =plt_x[j] + t_div_6/24


# plt_x[M1+M2+M3+M4+M5+M6] = t_1+t_2+t_3+t_4+t_5+t_6
# for j in range(M1+M2+M3+M4+M5+M6,M1+M2+M3+M4+M5+M6+M7-1):
#     plt_x[j+1] =plt_x[j] + t_div_7/24


# plt_x[M1+M2+M3+M4+M5+M6+M7] = t_1+t_2+t_3+t_4+t_5+t_6+t_7


# 1. Ave temp along tpct vs time 
# plt.figure(figsize=(12, 6), dpi=480)
# plt.plot(plt_x, np.average(nodes_u[l_adia:l_evap_end,:,1]), markersize= 12)
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('Average T along TPCT wall', fontsize=18)


# # 2. 
# plt.figure(figsize=(12, 6), dpi=480)
# plt.plot(plt_x, nodes_u[60,M,:], markersize= 12, label = 'Final time step')
# plt.plot(plt_x, nodes_u[60,0,:], markersize= 12, label='Initial conditions')
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('T at 60 [m]', fontsize=18)

# plt.figure(figsize=(12, 6), dpi=480)
# plt.plot(plt_x, nodes_u[90,M,:], markersize= 12, label = 'Final time step')
# plt.plot(plt_x, nodes_u[90,0,:], markersize= 12, label='Initial conditions')
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('T at 90 [m]', fontsize=18)

# plt.figure(figsize=(12, 6), dpi=480)
# plt.plot(plt_x, nodes_u[120,M,:], markersize= 12, label = 'Final time step')
# plt.plot(plt_x, nodes_u[120,0,:], markersize= 12, label='Initial conditions')
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('T at 120 [m]', fontsize=18)

# plt.figure(figsize=(12, 6), dpi=480)
# plt.plot(plt_x, nodes_u[150,M,:], markersize= 12, label = 'Final time step')
# plt.plot(plt_x, nodes_u[150,0,:], markersize= 12, label='Initial conditions')
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('T at 150 [m]', fontsize=18)

# # 3. Qt 

# plt.figure(figsize=(12, 6), dpi=480)
# plt.plot(plt_x[1:], Qt[1:], markersize= 12)
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('Output heat [W]', fontsize=18)









# --------------- STEADY STATE CHECK -----------------------------
# plt.figure(figsize=(12, 6), dpi=480)
# bd_cheq = np.zeros(N+1)
# for n in range (0,N+1):
#     bd_cheq[n] = np.average(nodes_u[l_adia+1:l_evap_end, M,n])
    
# bd_cheq_init = np.zeros(N+1)
# for z in range (0,N+1):
#     bd_cheq_init[z] = np.average(nodes_u[l_adia+1:l_evap_end, 0,z])
    
# bd_cheq_mid = np.zeros(N+1)
# for mm in range (0,N+1):
#     bd_cheq_mid[mm] = np.average(nodes_u[l_adia+1:l_evap_end, M - 500,mm])
    


# bd_cheq_r = np.zeros(N+1)
# for mm in range (0,N+1):
#     bd_cheq_r[mm] = np.average(nodes_u[l_adia+1:l_evap_end, 5320,mm])    
#plot temperature with the location -> index is from [1]
# plt.plot(delta[1:],bd_cheq_init[1:], label ='Initial')
# plt.plot(delta[1:],bd_cheq[1:], label ='Final time')
# # plt.plot(delta[1:],bd_cheq_mid[1:], label ='M - 500')
# # plt.plot(delta[1:],bd_cheq_r[1:], label ='Close to final time')
# plt.ylim([430 , 455])
# plt.xlabel('Location [m]', fontsize=12)
# plt.ylabel('Average Ground Temperature [K] ', fontsize=12)
# plt.legend()


# print ("last point: [K] ", np.average(nodes_u[l_adia+1:l_evap_end, 0,-1]) - np.average(nodes_u[l_adia+1:l_evap_end, M,-1]))
# #print("last 10 meters: ", )






# plt_x = np.zeros(M+1)
# plt_x[0] = 0
# for i in range(0,M1-1,1):
#     plt_x[i+1] = plt_x[i] + t_div_1/24
    
    
# plt_x[M1] = t_1
# for j in range(M1,M1+M2-1,1):
#     plt_x[j+1] =plt_x[j] + t_div_2/24


# plt_x[M1+M2] = t_1+t_2
# for j in range(M1+M2,M1+M2+M3-1,1):
#     plt_x[j+1] =plt_x[j] + t_div_3/24
    
# plt_x[M1+M2+M3] = t_1+t_2+t_3
# for j in range(M1+M2+M3,M1+M2+M3+M4-1,1):
#     plt_x[j+1] =plt_x[j] + t_div_4/24

# plt_x[M1+M2+M3+M4] = t_1+t_2+t_3+t_4
# for j in range(M1+M2+M3+M4,M1+M2+M3+M4+M5-1,1):
#     plt_x[j+1] =plt_x[j] + t_div_5/24


# plt_x[M1+M2+M3+M4+M5] = t_1+t_2+t_3+t_4+t_5
# for j in range(M1+M2+M3+M4+M5,M1+M2+M3+M4+M5+M6-1,1):
#     plt_x[j+1] =plt_x[j] + t_div_6/24


# plt_x[M1+M2+M3+M4+M5+M6] = t_1+t_2+t_3+t_4+t_5+t_6
# for j in range(M1+M2+M3+M4+M5+M6,M1+M2+M3+M4+M5+M6+M7-1,1):
#     plt_x[j+1] =plt_x[j] + t_div_7/24

# plt_x[M1+M2+M3+M4+M5+M6+M7] = t_1+t_2+t_3+t_4+t_5+t_6+t_7

# plt.figure(figsize=(12, 6), dpi=480)
# plt.plot(plt_x[10:], Qt[10:], markersize= 12)
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('Output heat [W]', fontsize=18)
# plt.plot(plt_x[10:], Qt[10:])
# plt.xlim([0, 1800])


# plt.figure(figsize=(12, 6), dpi=480)
# plt.plot(plt_x[11:], Resis[10:], markersize= 12)
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('Thermal Resistance [K/W]', fontsize=18)
# plt.plot(plt_x[11:], Resis[10:])
# plt.xlim([0, 1800])


# plt.figure(figsize=(12, 6), dpi=480)
# plt.plot(plt_x[10:], np.average(nodes_u[l_adia:l_evap_end,plt_x[10:],1]), markersize= 12)
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('Average Temperature [K]', fontsize=18)
# plt.plot(plt_x[10:], np.average(nodes_u[l_adia:l_evap_end,plt_x[10:],1]))
# plt.xlim([0, 1800])


#average temp
# abd = np.zeros(4063)

# for kek in range(0, 4063,1): abd[kek] = np.average(nodes_u[l_adia:l_evap_end, kek, 1])

# plt.figure(figsize=(12, 6), dpi=480)
# plt.plot(plt_x[10:], abd, markersize= 12)
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('Average Temperature [K]', fontsize=18)
# plt.plot(plt_x[10:], abd)
# plt.xlim([0, 1800])


elapsed_longterm = time.time() - t_longterm
# print("Running time [in sec] is:", elapsed_longterm)
# plt.figure(figsize=(12, 6), dpi=480)
# # plt.plot(delta[:], nodes_u[25, M, 1:], label="1M")
# plt.plot(delta[:], nodes_u[25, 300, 1:], label="15days")
# plt.plot(delta[:], nodes_u[25, 0, 1:], label="Initial conditions")
# plt.xlabel('Location [m]', fontsize=18)
# plt.ylabel('Temperature [K]', fontsize=18)
# plt.legend()
# #plt.ylim([340, 374])
# nodes_u[25, M, -1] - nodes_u[25, 0, -1]




'''
some comments based on debugging:
    
1. When the output power values are out of range -> probably the profile is increasing:
    the reason is that somewhere in time, the temperature of the ground became bigger than the
    average temperature in the ground --> in automative TPCT (like Ahmad work) TPCT works at certain 
    temperature not any, and you have not put something as warning or stopping criteria in the code for this 
    conditions yet. (10 January)



2. What ?



'''


# old temperature of air loop 
 # if k == 0:
 #     T_air[day] = 282.587   
 # if 0< k <= M1:
 #     if ((k)*t_div_1)%24 != 0:
 #         T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
 #     elif ((k)*t_div_1)%24 == 0:
 #         day = day +1
 #         #print(day)
 #         T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771)  
 
 # if   M1 < k <= M2:  
 #     if ((k)*t_div_2)%24 != 0:
 #           T_air= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
 #     elif ((k)*t_div_2)%24 == 0:
 #         day = day +1
 #         T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771)  

 # if   M2 < k <= M3:  
 #     if ((k)*t_div_3)%24 != 0:
 #             T_air= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
 #     elif ((k)*t_div_3)%24 == 0:
 #         day = day +1
 #         T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771)  
 
 # if   M3 < k <= M4:         
 #     if ((k)*t_div_4)%24 != 0:
 #         T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
 #     elif ((k)*t_div_4)%24 == 0:
 #         day = day +1
 #         T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771)        
  
 # if   M4 < k <= M5:  
 #       if ((k)*t_div_5)%24 != 0:
 #           T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
 #       elif ((k)*t_div_5)%24 == 0:
 #           day = day +1
 #           T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771)  
 # if   M6 < k <= M7:  
 #     if ((k)*t_div_6)%24 != 0:
 #         T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
 #     elif ((k)*t_div_6)%24 == 0:
 #         day = day +1
 #         T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771)  


 # if   M7 < k:          
 #     if ((k)*t_div_7)%24 != 0:
 #         T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771) 
 #     elif ((k)*t_div_7)%24 == 0:
 #         day = day +1
 #         T_air[day]= 273.15 +16.14 -6.703*np.cos(day*0.01771 ) -7.57*np.sin(day*0.01771)  











