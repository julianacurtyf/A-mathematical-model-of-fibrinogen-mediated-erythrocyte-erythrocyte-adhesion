#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:23:48 2021

@author: julianacurtyf@gmail.com
"""

from numpy import sqrt
from matplotlib.pyplot import figure, plot, savefig, legend, title
from time import time
from rbc import k_matrices, volume, area, gradient, laplacian, time_domain, frequency_domain, sliced_figure, save_data
from utils import *

# Variables definition

ini = time()

nx = 85; ny = 85; nz = 60

gama = -0.03; eta = -0.06
beta = -1; eps = 1.0
alpha = 30

vel_phi = 0.05; vel_psi = -0.05

dt = 0.005

B = 2e-18
eps_real = 1.7e-7

v_f1 = []; v_f2 = []; v_f3 = []

# Matrix definition

name = 'phi_d_18_eta_' + str(eta) + '_gama' + str(gama)

phi = get_data('phi_d_18')
psi = get_data('psi_d_18')

Vt_phi = volume(phi); Vt_psi = volume(psi)

k1,k2,k3,k = k_matrices(nx,ny,nz)


for it in range(20000):
    
    if it == 0: # Initial moment
    
        save_vti_file(phi, psi, nx, ny, nz, str(it)+'_rbc'+name)
                
    if it == 4000: # RBCs already touching
    
        save_vti_file(phi, psi, nx, ny, nz, str(it)+'_rbc'+name)
        
    if it == 8000: # Change movement direction
        
        vel_psi = -vel_psi
        vel_phi = -vel_phi
        
        save_vti_file(phi, psi, nx, ny, nz, str(it)+'_rbc'+name)
    
    if it == 10000: # Cells starting separating
    
        save_vti_file(phi, psi, nx, ny, nz, str(it)+'_rbc'+name)
    
    if it == 20000: # Cells starting separating
    
        save_vti_file(phi, psi, nx, ny, nz, str(it)+'_rbc'+name)
            
    if it == 30000: # Cells in the middle of separating
    
        save_vti_file(phi, psi, nx, ny, nz, str(it)+'_rbc'+name)
    
    if it == 35000: # Cells separated
    
        save_vti_file(phi, psi, nx, ny, nz, str(it)+'_rbc'+name)

    if it == 40000: # End of separation
    
        save_vti_file(phi, psi, nx, ny, nz, str(it)+'_rbc'+name)
       
    grad_phi_1 = gradient(k1,phi)
    grad_phi_2 = gradient(k2,phi)
    grad_phi_3 = gradient(k3,phi)
    
    grad_psi_1 = gradient(k1,psi)
    grad_psi_2 = gradient(k2,psi)
    grad_psi_3 = gradient(k3,psi)
    
    laplac_phi = laplacian(k, phi)
    laplac_psi = laplacian(k, psi)
    
    phi_k = frequency_domain(phi)
    psi_k = frequency_domain(psi)
    
    h_phi = h_matrix(phi)
    h_psi = h_matrix(psi)
    
    laplac_h_phi = laplacian(k,h_phi)
    laplac_h_psi = laplacian(k,h_psi)
    
    force = -gama*(1-phi**2)*h_psi + eta*(1-phi**2)*laplac_h_psi
    
    grad_force_1 = gradient(k1,force)
    grad_force_2 = gradient(k2,force)
    grad_force_3 = gradient(k3,force)
     
    area_phi = area(grad_phi_1,grad_phi_2,grad_phi_3,eps)
    area_psi = area(grad_psi_1,grad_psi_2,grad_psi_3,eps)
    
    volume_phi = volume(phi)
    volume_psi = volume(psi)
    
    if it == 0:
        At_phi = area(grad_phi_1,grad_phi_2,grad_phi_3,eps)
        At_psi = area(grad_psi_1,grad_psi_2,grad_psi_3,eps)
    
    nlin_phi = (3*phi**2 - 1)*(phi**3 - phi - laplac_phi) + beta*(1-phi**2)*(Vt_phi-volume_phi)/Vt_phi +\
        3/sqrt(2)*alpha*eps*laplac_phi*(At_phi-area_phi)/At_phi + (1-phi**2)*(-gama*h_psi+eta*laplac_psi) - vel_phi*grad_phi_3
    nlin_psi = (3*psi**2 - 1)*(psi**3 - psi - laplac_psi) + beta*(1-psi**2)*(Vt_psi-volume_psi)/Vt_psi +\
        3/sqrt(2)*alpha*eps*laplac_psi*(At_psi-area_psi)/At_psi + (1-psi**2)*(-gama*h_phi+eta*laplac_phi) - vel_psi*grad_psi_3
    
    nlin_phi_k = frequency_domain(nlin_phi)
    nlin_psi_k = frequency_domain(nlin_psi)
     
    phi_sc_k = frequency_domain(phi**3 - phi)
    psi_sc_k = frequency_domain(psi**3 - psi)
    
    phi_novo_k = (phi_k - 2*dt*(nlin_phi_k + k**2*eps**2*phi_sc_k))/(1+2*dt*k**4)
    psi_novo_k = (psi_k - 2*dt*(nlin_psi_k + k**2*eps**2*psi_sc_k))/(1+2*dt*k**4)
   
    f1 = rbc_force(phi, grad_force_1, B, eps_real)
    f2 = rbc_force(phi, grad_force_2, B, eps_real)
    f3 = rbc_force(phi, grad_force_3, B, eps_real)
    
    v_f1.append(f1.real)
    v_f2.append(f2.real)
    v_f3.append(f3.real)
    
    phi = time_domain(phi_novo_k)
    psi = time_domain(psi_novo_k)
    
    if it%1000  == 0 :
        sliced_figure(phi+psi,40,str(it)+name)

figure(1)
plot(v_f1, label = 'x axis')
plot(v_f2, label = 'y axis')
plot(v_f3, label = 'z axis')
legend()
title('Forces (N)')
savefig('force'+name+'.png', dpi = 1000)

save_data(v_f1,'f1'+name)
save_data(v_f2,'f2'+name)
save_data(v_f3,'f3'+name)

fim = time()

print((fim-ini)/60)


