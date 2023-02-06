# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:39:53 2022

@author: Mohammad
"""
# Thermal resistance for the Experiment 

import numpy as np
Te_o = 273.15 + 20  # Temperature of the evaporator 
Tc_ave = 273.15 -10 # Temperature of the coolant @ shell and tube 

k_steel = 45  #WMk
le = 0.75 #m      To change the areas, just change this parameter
Do_e = 1.35 # inch 
Di_e = 1  # inch 

# Known resistance are as below: 
    
# Resistance @ evaporator with the length of "le"
R1 = 1/(2*np.pi*k_steel*le) *np.log(Do_e/Di_e) 

# Resistance @ condenser with the length of "ln"
ln = 1 - le
R4 = 1/(2*np.pi*k_steel*ln) *np.log(Do_e/Di_e) 

# Resistance outside the condenser with the same length of the condenser 
# Nusselt for laminar coaxial flow (eq 29. paper: Hirbodi et al 2022 on laminar correlations for coaxial pipes)
Di_shell = 2.2 #inch
r_star = Do_e/Di_shell
Nu = 0.791*r_star**1.072 + 0.542*r_star**0.179+2.667
h_5= k_steel*Nu/ln
R5 = 1/(h_5 * np.pi*(Di_shell*2.5/100)*ln)


# Newton-Raphson is required to find R2 and R3 through an iterative process 
Q = np.zeros(2)
Q[0] = 1000 #initial guess 

#R744 (liquid CO2) -- 
rho_htf_l =  1100 
k_htf_l = 0.16
cp_htf_l = 1000  #j/kg.k    
g = 9.8
r_in = Di_e/2
rho_htf_v = 1.3

# Check and finilize the numbers 
h_htf_lv = 210000 #j/kg
mu_htf_l = 0.0002
P_s =  35    # saturation pressure @ ave temperature 
P_atm =    1  # atmospheric pressure 
lt = ln + le


for k in range (0,1,1):
    def TotalHeat(Q):
            zi = 0.32*(rho_htf_l**0.65*k_htf_l**0.3*cp_htf_l**0.7*g**0.2*(Q/(2*np.pi*r_in*le))**0.4)/(rho_htf_v**0.25*h_htf_lv**0.4*mu_htf_l**0.1)
            h_2 = zi*(P_s/P_atm)**0.3
            h_3 = 0.925*((k_htf_l**3*rho_htf_l**2*g*h_htf_lv)/(mu_htf_l*(Q/(2*np.pi*r_in*ln))*lt))**(1/3)  #phase change
            Req = R1 + 1/(h_2 * (2*np.pi*r_in*le)) + 1/(h_3* (2*np.pi*r_in*ln)) + +R4 + R5
            return   Q - abs(Tc_ave - Te_o)/(Req)
        
            
    def derivative(f, Q,dh):
        return (f(Q+dh)-f(Q-dh)) / (2.0*dh)
          
            
    def solve(f, Q0,dh):
        lastX = Q0
        nextX = lastX + 100 *dh 
        while (abs(lastX - nextX) >dh): 
            newY = f(nextX)
            lastX = nextX
            nextX = lastX - newY / derivative(f, lastX,dh) 
        return nextX
    
    
    QFound = solve(TotalHeat, Q[k],0.1) 
    Q[k+1] = QFound.copy()  
    zi = 0.32*(rho_htf_l**0.65*k_htf_l**0.3*cp_htf_l**0.7*g**0.2*(Q[k+1]/(2*np.pi*r_in*le))**0.4)/(rho_htf_v**0.25*h_htf_lv**0.4*mu_htf_l**0.1)
    h_2 = zi*(P_s/P_atm)**0.3
    h_3 =  0.925*(k_htf_l**3*rho_htf_l**2*g*h_htf_lv)/(mu_htf_l*(Q[k+1]/(2*np.pi*r_in*ln))*ln)**0.3333
    
    
    
A_evap = 2 * np.pi * r_in*le # this is the perimeter of the evaporator times length section   
R2 = 1/(h_2 * A_evap)
A_cond = 2 * np.pi * r_in*ln   # assumed the same radius 
R3 = 1 / (h_3 * A_cond)

Tstar = Te_o - Q[-1]*(R1 + R2)  
Tstar_dnward =  Tc_ave  + Q[-1]*(R3 + R4 + R5)  
print("Total Heat Extracted is: ")       
print(Q[-1],  "[W]")
print("---------")  
print("Temperature inside the TPCT, referemce point is EVAPORATOR: ")  
print(Tstar,  "[K]")
print("---------") 
print("Temperature inside the TPCT, referemce point is CONDENSER: ")  
print(Tstar_dnward,  "[K]")

Q[-1]/A_evap
Q[-1]/A_cond
    

import matplotlib.pyplot as plt


x = [3/8 , 3/4 , 1]  # Diameter 
y = [291, 291.7 , 291.94]  #temperature 
#z = [285.9/A_evap, 336.5/A_evap, 359.7/A_evap]
z = [646, 380, 305]
plt.xlabel("Diameter in [in]")
plt.ylabel("Temerature in [k] inside the TPCT")
#plt.ylabel("Heat Flux at the Evaporator $[W/m^2]$")
plt.plot(x, y, 'o-')
#plt.plot(x, z, 'ro-')
































