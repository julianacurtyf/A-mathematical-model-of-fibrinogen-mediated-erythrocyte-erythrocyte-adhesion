#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:25:12 2021

This program initialises the two cells

@author: julianacurtyf@gmail.com
"""

from numpy import sqrt
from matplotlib.pyplot import plot
from utils import *
    

# Variables definition

lx = 85; ly = 85; lz = 60
dx = 1; dy = 1; dz = 1  

x_phi = 39; y_phi = 32; z_phi = 42
x_psi = 39; y_psi = 50; z_psi = 25

angle = 0
a = 26; b = 26; c = 7

nt = 100001; dt = 0.005
beta = -1; eps = 1

alpha = 0 # alpha starts being zero and area conservation only kicks in after 500 time steps,  when interface is formed

At_phi = 1; At_psi = 1

evolut_area = []
evolut_volum = []

# Matrix definition

phi = phi_matrix(lx, ly, lz, dx, dy, dz)
psi = phi_matrix(lx, ly, lz, dx, dy, dz)

sigma_phi = sigma_matrix(lx, ly, lz)
sigma_psi = sigma_matrix(lx, ly, lz)

phi = ellipsoid(phi, x_phi, y_phi, z_phi, angle, a, b, c)
psi = ellipsoid(psi, x_psi, y_psi, z_psi, angle, a, b, c)
            
phi_novo = phi
psi_novo = psi

k1, k2, k3, k = k_matrices(lx, ly, lz)

Vt_phi = volume(phi)  
Vt_psi = volume(psi)


for it in range(nt):
    
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
    
    if it == 500: 
        alpha = 30;
        At_phi = area(grad_phi_1,grad_phi_2,grad_phi_3,eps)
        At_psi = area(grad_psi_1,grad_psi_2,grad_psi_3,eps)
    
    area_phi = area(grad_phi_1,grad_phi_2,grad_phi_3,eps)
    area_psi = area(grad_psi_1,grad_psi_2,grad_psi_3,eps)
    evolut_area.append(area_phi)
    
    volume_phi = volume(phi) 
    volume_psi = volume(psi) 
    evolut_volum.append(volume_phi)
    
    nlin_phi = (3*phi**2 - 1)*(phi**3 - phi - eps**2*laplac_phi) + beta*(1-phi**2)*(Vt_phi-volume_phi)/Vt_phi + 3/sqrt(2)*alpha*eps*laplac_phi*(At_phi-area_phi)/At_phi
    nlin_psi = (3*psi**2 - 1)*(psi**3 - psi - eps**2*laplac_psi) + beta*(1-psi**2)*(Vt_psi-volume_psi)/Vt_psi + 3/sqrt(2)*alpha*eps*laplac_psi*(At_psi-area_psi)/At_psi
    
    nlin_phi_k = frequency_domain(nlin_phi)
    nlin_psi_k = frequency_domain(nlin_psi)
    
    phi_sc_k = frequency_domain(phi**3 - phi)
    psi_sc_k = frequency_domain(psi**3 - psi) 
    
    phi_novo_k = (phi_k - 2*dt*(nlin_phi_k + k**2*eps**2*phi_sc_k))/(1+2*dt*k**4)
    psi_novo_k = (psi_k - 2*dt*(nlin_psi_k + k**2*eps**2*psi_sc_k))/(1+2*dt*k**4)
    
    phi = time_domain(phi_novo_k)
    psi = time_domain(psi_novo_k)
    
    if it%1000 == 0:
        sliced_figure(phi,40,'1')
        sliced_figure(psi,40,'1')


plot(evolut_area)
plot(evolut_volum)

save_data(phi, 'phi_d_18')

save_data(psi, 'psi_d_18')

