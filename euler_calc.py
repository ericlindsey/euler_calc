# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:14:01 2016

@author: elindsey
"""

import numpy as np
import scipy.linalg
import geod_transform
#import gps_data

def identify_block(GPS,blockids,sig_thres=1,misfit_thres=1.5,maxiter=30):
    '''Given a GPS dataset and an initial set of stations, find stations consistent
    with the estimated block. Each iteration re-evaluates the fit to all stations
    and re-estimates the block using only stations satisfying the uncertainty and
    misfit criteria.'''
    assert len(blockids)>1
    changed_stations=True
    numiter=0
    while changed_stations:
        numiter=numiter+1
        blockpole,cov_pole,chi2red=best_fit_pole(GPS.lat[blockids],GPS.lon[blockids],GPS.vE[blockids],GPS.vN[blockids],
                                                            GPS.sigE[blockids],GPS.sigN[blockids],GPS.rhoEN[blockids])
        print(blockpole,chi2red)
        # compute misfit for each station
        pred_e,pred_n=pole_velocity(GPS.lat,GPS.lon,blockpole[0],blockpole[1],blockpole[2]) 
        misfits=GPS.calc_point_misfit(pred_e,pred_n)
        #check for any changed stations        
        old_blockids=blockids
        blockids=np.where(np.logical_and(GPS.sigE<sig_thres, np.logical_and(GPS.sigN<sig_thres, misfits<misfit_thres) ))
        if np.array_equal(blockids,old_blockids) or numiter>maxiter:
            changed_stations=False
            print("finishing after %d iterations"%numiter)
    return blockids,blockpole

def best_fit_pole(lat,lon,ve,vn,sige,sign,rho):
    '''Given lists of lat,lon, and east,north velocity (and uncertainties), find
    the best-fitting euler pole and its associated uncertainties and misfit statistics.'''
    ## preparation    
    #construct the observation vector (2nx1)
    vel=np.ravel(np.column_stack((ve,vn)))
    #construct the 2nx2n covariance matrix (2nx2n)
    cov=get_2d_covar_mat(sige,sign,rho)
    #construct the cross product design matrix (2nx3)
    Rx=euler_rot_matrix(lat,lon)
    
    ## computation
    #do the matrix inversion - using the basic (weighted) Normal equations
    n_inv=scipy.linalg.inv(np.dot(Rx.T,np.dot(cov,Rx)))
    fitpole=np.dot(n_inv,np.dot(Rx.T,np.dot(cov,vel)))
    #convert the euler vector to a geodetic location and rotation rate    
    latp,lonp,degmyr=euler_location(fitpole)
    
    ## uncertainties
    v_predict_e,v_predict_n=pole_velocity(lat,lon,latp,lonp,degmyr)
    v_resid_e=ve - v_predict_e
    v_resid_n=vn - v_predict_n
    v_resid=np.ravel(np.column_stack((v_resid_e,v_resid_n)))
    dof=2*np.size(ve)-3
    sigma_0=np.sqrt(np.dot(v_resid.T,np.dot(cov,v_resid))/dof)
    cov_pole_cart=sigma_0**2 * n_inv
    J=euler_jacobian([latp,lonp,degmyr])
    cov_pole_geod=np.dot(J,np.dot(cov_pole_cart,J.T))
    
    #chi squared/dof statistic (reduced chi squared).
    #Note this ignores rho! But typically rho is small for most stations.
    uncert_inv=np.ravel(np.column_stack((sige**-2,sign**-2)))
    chi2_red=np.dot(v_resid**2,uncert_inv)/dof
    
    return np.array([latp,lonp,degmyr]),cov_pole_geod,chi2_red

def pole_velocity(lat,lon,latp,lonp,degmyr):
    '''list,list,scalar,scalar,scalar > list,list
    Returns predicted horizontal (e,n) velocities (in mm/yr) at each location.
    Inputs are a list of coordinates (lat,lon in degrees) and one euler pole
    (lat,lon,rate(in deg/myr)).    
    Geodetic (ellipsoid) coordinates in degrees are assumed but the rotation is
    done assuming a spherical earth.''' 
    #convert pole to a cartesian vector
    omega=euler_vector(latp,lonp,degmyr) 
    #rotation matrix accomplishes the cross product
    Rx=euler_rot_matrix(lat,lon)
    vout=np.dot(Rx,omega)
    #separate result into east, north components
    return vout[0::2],vout[1::2]
    
def euler_vector(latp,lonp,degmyr):
    '''Convert geodetic euler pole (lat,lon,deg/myr) to a scaled ECEF rotation vector,
    scaled to give units of mm/yr when multiplied by a rotation matrix based on
    spherical earth coordinates.'''
    latpc=geod_transform.geod2spher(latp)
    #convert to radians, then use simple spherical coordinate conversion
    latr=np.radians(latpc)
    lonr=np.radians(lonp)
    px=np.cos(lonr)*np.cos(latr)
    py=np.sin(lonr)*np.cos(latr)
    pz=np.sin(latr)
    # return a rescaled value that works when multiplied by coordinates of unit magnitude
    # (assumed on the sphere with earth's radius)
    return 6371. * degmyr * (np.pi/180.) * np.array([px,py,pz])

def euler_location(omega):
    ''' convert cartesian rotation vector (3x1) [px,py,pz] to pole location (lat,lon,degmyr)'''
    degmyr=np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)/(6371.*np.pi/180.)
    latr=np.arctan(omega[2]/np.sqrt(omega[0]**2 + omega[1]**2))
    lat=geod_transform.spher2geod(np.degrees(latr))
    lon=np.degrees(np.arctan2(omega[1],omega[0]))
    return lat,lon,degmyr   
    
def euler_jacobian(omega):
    ''' get jacobian matrix of the euler pole for error analysis. Parameter order is lat,lon,rate.'''
    mag=np.sqrt(omega[0]**2+omega[1]**2+omega[2]**2)
    J=np.array([[(-1/mag**2)*(omega[0]*omega[2]/np.sqrt(omega[0]**2 + omega[1]**2)),
                (-1/mag**2)*(omega[1]*omega[2]/np.sqrt(omega[0]**2 + omega[1]**2)),
                (-1/mag**2)*(np.sqrt(omega[0]**2 + omega[1]**2))  ],
               [ -omega[1]/(omega[0]**2 + omega[1]**2), -omega[0]/(omega[0]**2 + omega[1]**2), 0 ],
               [omega[0]/mag,    omega[1]/mag,    omega[2]/mag]])
    return J

def get_2d_covar_mat(esigma,nsigma,rhoen):
    '''get block-diagonal 2d covariance matrix given 2 sigma values and a correlation'''
    esigma=np.array(esigma,ndmin=1)
    nsigma=np.array(nsigma,ndmin=1)
    rhoen=np.array(rhoen,ndmin=1)
    covarmat=None
    for i in range(np.size(esigma)):
        covi=np.array([[esigma[i]*esigma[i],          esigma[i]*nsigma[i]*rhoen[i]],
                       [esigma[i]*nsigma[i]*rhoen[i], nsigma[i]*nsigma[i]         ]])
        if covarmat is None:
            covarmat=covi
        else:
            covarmat=scipy.linalg.block_diag(covarmat,covi)
    return covarmat

def euler_rot_matrix(lat,lon):
    '''Given lists of cartesian (lat,lon) coordinates, form the matrix that will
    accomplish a cross product when multiplied into a 3x1 cartesian euler vector.
    For each point (x,y,z) there will be a 2x3 block:
    [[ -sin(lat)cos(lon) -sin(lat)cos(lon) cos(lat)]
     [  sin(lon)         -cos(lon)          0]]
    Thus, the final result has dimensions 2nx3.'''
    assert np.size(lat) == np.size(lon), "lat,lon vectors should be the same length."
    #convert to spherical latitude    
    latc=geod_transform.geod2spher(lat)
    #convert to radians, and ensure format is an array
    latr=np.array(np.radians(latc),ndmin=1)
    lonr=np.array(np.radians(lon),ndmin=1)
    #form the 2nx3 array
    Rx=np.empty((0,3))
    for i in range(np.size(latr)):
        Ri=np.array([[ -np.sin(latr[i])*np.cos(lonr[i]), -np.sin(latr[i])*np.sin(lonr[i]), np.cos(latr[i])],
                     [  np.sin(lonr[i])                 , -np.cos(lonr[i])                 , 0]])
        Rx = np.row_stack((Rx,Ri))
    return Rx

##end##
