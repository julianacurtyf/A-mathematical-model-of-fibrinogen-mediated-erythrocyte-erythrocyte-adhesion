#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:54:49 2023

@author: julianacurtyf@gmail.com
"""

from numpy import tanh, zeros, ones, sin, cos, pi, arange, sqrt, fft
from matplotlib.pyplot import imshow, show, figure, title, xlabel, ylabel, savefig
import pickle


def phi_matrix(lx, ly, lz, dx, dy, dz):
    """"Create a matrix phi
       
        Parameters
        ----------
            lx: integer
                x axis length
                
            ly: integer
                y axis length
                
            lz: float
                z axis length
            
            dx: integer
                distance between two points in x axis
                
            dy: integer
                distance between two points in y axis
            
            dz: integer
                distance between two points in z axis
                   
        Returns
        -------
        array
            matrix phi 
       """
    x = arange(0,lx,dx); y = arange(0,ly,dy); z = arange(0,lz,dz)
    nx = len(x); ny = len(y); nz = len(z)
    phi = ones((nx,ny,nz))*(-1)
    
    return phi


def sigma_matrix(nx,ny,nz):
    
    sigma = zeros((nx,ny,nz))

    return sigma


def k_matrices(nx,ny,nz):
    
    k1 = zeros((nx,ny,nz))
    k2 = zeros((nx,ny,nz))
    k3 = zeros((nx,ny,nz))
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                k1[:,iy,iz] = fft.fftfreq(nx)*(2*pi)
                k2[ix,:,iz] = fft.fftfreq(ny)*(2*pi)
                k3[ix,iy,:] = fft.fftfreq(nz)*(2*pi)
  
    k = sqrt(k1**2+k2**2+k3**2)
    
    return k1, k2, k3, k


def ellipsoid(phi, x_phi, y_phi, z_phi, angle, a, b, c):
    """"Create an ellipsoid in a matrix phi
       
        Parameters
        ----------
            phi: 3D array
            
            x_phi: float
                position of the center of the ellipsoid
                
            y_phi: float
                position of the center of the ellipsoid
                
            z_phi: float
                position of the center of the ellipsoid
                
            angle: float
                angle, in rad, reponsable for a rotation in the ellipsoid
            
            a: float
                ellipsoid's x axis
                
            b: float
                ellipsoid's y axis
            
            c: float
                ellipsoid's z axis
                   
        Returns
        -------
        array
            matrix phi 
       """
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz): # Cycle to create initial cell shape
                
                x_phi_rot = x_phi*cos(angle)-z_phi*sin(angle)
                z_phi_rot = x_phi*sin(angle)+z_phi*cos(angle)
                ix_rot = ix*cos(angle)-iz*sin(angle)
                iz_rot = ix*sin(angle)+iz*cos(angle)
                phi[ix,iy,iz] = tanh((-sqrt((ix_rot-x_phi_rot)**2 + (iy-y_phi)**2 + \
                   (iz_rot-z_phi_rot)**2/(c/b)**2)+30)/5)
                    
    return phi


def volume(phi):
    
    vol = sum(sum(sum((1+phi)/2))) 
    
    return vol
    

def area(grad1, grad2, grad3,eps):
    
    area = sum(sum(sum(3/(2*sqrt(2))*eps*(grad1**2+grad2**2+grad3**2))))
    
    return area


def gradient(k,phi):
    
    phi_k = frequency_domain(phi)
    
    grad_k = complex(0,1)*k*phi_k

    grad = time_domain(grad_k)
    
    return grad


def laplacian(k, phi):
    
    phi_k = frequency_domain(phi)
    
    laplac_k = -k**2*phi_k
    
    laplac = time_domain(laplac_k)

    return laplac


def frequency_domain(phi):
    
    phi_k = fft.fftn(phi)
    
    return phi_k


def time_domain(phi_k):
    
    phi = fft.ifftn(phi_k)
    
    return phi


def sliced_figure(phi,pos,name):
    phi = phi.real
    figure(1)
    title('Phi')
    xlabel('[\u03BCm]')
    #xticks(arange(0,90,15),[ 0.  ,  2.55,  5.1 ,  7.65, 10.2 , 12.75, 15.3 ])
    ylabel('[\u03BCm]')
    #yticks(arange(0,100,10),[ 0. ,  1.7,  3.4,  5.1,  6.8,  8.5, 10.2, 11.9, 13.6, 15.3])
    imshow(phi[pos,:,:],cmap='Reds')
    savefig(name+'.png', dpi = 1000)
    show()
    
 
def save_data(phi, name):
    
    f = open(name +'.pckl', 'wb')
    pickle.dump(phi, f)
    f.close()
 

def get_data(name):

    f = open(name+'.pckl', 'rb')
    phi = pickle.load(f)
    f.close()
    
    return phi


def save_vti_file(phi, psi, nx, ny, nz, name):
    
    pc_real = phi.real + psi.real
    pc_lista_novo = []
    
    for iz in range(nz-1):
        for iy in range(ny-1):
            for ix in range(nx-1):
                t = pc_real[ix,iy,iz]
                pc_lista_novo.append(t)
    
    pc_string_novo = "    ".join([str(_) for _ in pc_lista_novo]) # criacao de uma string com os valores da lista
    
    with open(name + ".vti", "w" ) as my_file:
        my_file.write('<?xml version="1.0"?>')
        my_file.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        my_file.write('  <ImageData WholeExtent="0 '+str(nx-1)+' 0 '+str(ny-1)+' 0 '+str(nz-1)+'" Origin="0 0 0" Spacing ="1 1 1">\n')
        my_file.write('    <Piece Extent="0 '+str(nx-1)+' 0 '+str(ny-1)+' 0 '+str(nz-1)+'">\n') # dimensao da matriz x1 x2 y1 y2 z1 z2
        my_file.write('     <CellData>\n')
        my_file.write('     <DataArray Name="scalar_data" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo)
        my_file.write('\n         </DataArray>\n')
        my_file.write('      </CellData>\n')
        my_file.write('    </Piece>\n')
        my_file.write('</ImageData>\n')
        my_file.write('</VTKFile>\n')
        my_file.close()


def rbc_force(phi, grad, B, eps):
    
    force = -3*sqrt(2)*B/8/eps*sum(sum(sum((phi+1)/2*grad)))
    
    return force


def h_matrix(phi):
    
    h = (1+phi)**2*(2-phi)
    
    return h
    
