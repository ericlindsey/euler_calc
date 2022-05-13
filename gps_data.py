# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:42:38 2016

@author: elindsey
"""

#module for GPS data; an object instance is a dataset

# properties
#lat,lon,vE,vN,vU,sigE,sigN,sigU,rhoEN,name,type
#other? timespan, reference?

# methods:
#load_2d_data
#(load_3d_data)
#calc_residual (fit to a model; just subtracts)
#calc_misfit (use covariance matrix to get the total misfit, for one or many stations)

import numpy as np
import scipy.linalg #, scipy.sparse, scipy.sparse.linalg
import geod_transform

class GPSData:
    
    #    def __init__(self):
    #        self.lon=np.array([])
    #        self.lat=np.array([])
    #        self.vE=np.array([])
    #        self.vN=np.array([])
    #        self.vU=np.array([])
    #        self.sigE=np.array([])
    #        self.sigN=np.array([])
    #        self.sigU=np.array([])
    #        self.rhoEN=np.array([])
    #        self.cov=np.array([])
    #        self.inv_cov=np.array([])
    #        self.velvec=np.array([])
    #        self.name=[]
    #        self.reference=[]
    #        self.type=[]
    #        self.is2d=False
    #        
    def load_2d_data(self,fname,order=None):
        #default column order is [lon lat vE vN sigE sigN rho SITE Reference]
        # if another format is used, pass an array to 'order' argument with the index to locations of those 8 columns.
        # if sites have no name or reference, omit or use 'np.NaN' for the final columns.
        #
        # read column order
        if order is None:
            order=[0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.is2d=True
        #read data
        dat=np.genfromtxt(fname)
        self.lon=dat[:,order[0]]
        #use positive longitudes
        self.lon=np.where(self.lon>0,self.lon,self.lon+360)
        self.lat=dat[:,order[1]]
        self.vE=dat[:,order[2]]
        self.vN=dat[:,order[3]]
        self.sigE=dat[:,order[4]]
        self.sigN=dat[:,order[5]]
        self.rhoEN=dat[:,order[6]]
        self.velvec=np.ravel(np.column_stack((self.vE,self.vN)))
        #read filenames
        if len(order)>7 and not np.isnan(order[7]):
            self.name=np.genfromtxt(fname,usecols=order[7],dtype='str')
        #read references
        if len(order)>8 and not np.isnan(order[8]):
            self.reference=np.genfromtxt(fname,usecols=order[8],dtype='str')
        #compute covariance matrix and its inverse
        self.calc_cov()
        
    def create_2d_dataset(self,data):
        # create a GPS data object from pre-loaded data
        # bare-bones object, does not include names or references.
        # assumed column order is [lon lat vE vN sigE sigN rho]
        self.is2d=True
        self.lon=data[:,0]
        self.lat=data[:,1]
        self.vE=data[:,2]
        self.vN=data[:,3]
        self.sigE=data[:,4]
        self.sigN=data[:,5]
        self.rho=data[:,6]
        #use positive longitudes
        self.lon=np.where(self.lon>0,self.lon,self.lon+360)
        #create 2nx1 vector
        self.velvec=np.ravel(np.column_stack((self.vE,self.vN)))
        #compute covariance matrix and its inverse
        self.calc_cov()
                
    def load_3d_data(self,fname,order=None):
        #default column order is [lon lat vE vN sigE sigN rho vU sigU SITE]
        # if another format is used, pass an array to 'order' argument with the index to locations of those 8 columns.
        # if sites have no name, use 'np.NaN' for the final column.
        #
        # read column order
        if order is None:
            order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.is2d=False
        #read data
        dat=np.genfromtxt(fname)
        self.lon=dat[:,order[0]]
        #use positive longitudes
        self.lon=np.where(self.lon>0,self.lon,self.lon+360)
        self.lat=dat[:,order[1]]
        self.vE=dat[:,order[2]]
        self.vN=dat[:,order[3]]
        self.sigE=dat[:,order[4]]
        self.sigN=dat[:,order[5]]
        self.rhoEN=dat[:,order[6]]
        self.vU=dat[:,order[7]]
        self.sigU=dat[:,order[8]]
        self.velvec=np.ravel(np.column_stack((self.vE,self.vN,self.vU)))
        #compute covariance matrix and its inverse
        self.calc_cov()
        
        #read filenames
        if not np.isnan(order[9]):
            self.name=np.genfromtxt(fname,usecols=order[9],dtype='str')
            
    def deduplicate_data(self,maxdist=500):
        #sometimes datasets contain multiple solutions for the same station.
        #In this case, although we could use both, for e.g. plotting purposes we only 
        #want to show the one with smaller uncertainties.
        idelete=np.array([])
        for i in range(len(self.lon)-1):
            duprow=np.zeros(len(self.lon))
            distrow=geod_transform.haversine(self.lat[i],self.lon[i],self.lat[i+1:],self.lon[i+1:])
            duprow[i+1:]=distrow<maxdist          
            if max(duprow)>0:
                icheck=np.where(duprow>0)[0]
                for j in icheck:
                    if self.sigE[j]**2 + self.sigN[j]**2 >= self.sigE[i]**2 + self.sigN[i]**2:
                        idelete=np.append(idelete,j)
                    else:
                        idelete=np.append(idelete,i)
        idelete=np.unique(idelete)
        self.delete_station(idelete)           
        print('Deleted %d duplicate stations' %len(idelete))
        
    def delete_station(self,i):
        self.lat = np.delete(self.lat,i)
        self.lon = np.delete(self.lon,i)
        self.vE = np.delete(self.vE,i)
        self.vN = np.delete(self.vN,i)
        self.sigE = np.delete(self.sigE,i)
        self.sigN = np.delete(self.sigN,i)
        self.rhoEN = np.delete(self.rhoEN,i)
        if self.is2d:
            idel=[2*i,2*i+1]
        else:
            idel=[3*i,3*i+1,3*i+2]
            self.vU = np.delete(self.vU,i)
            self.sigU = np.delete(self.sigU,i)
        self.velvec = np.delete(self.velvec,idel)
        self.name = np.delete(self.name,i)
        self.cov  = np.delete(self.cov,i,axis=0)
        self.inv_cov  = np.delete(self.inv_cov,i,axis=0)

    def calc_misfit(self,vEmodel,vNmodel,vUmodel=None):
        #total misfit statistic. Take exp(-misfit/2) for (something proportional to) likelihood
        misfits=self.calc_point_misfit(vEmodel,vNmodel,vUmodel)
        return sum(misfits)
        
    def calc_point_misfit(self,vEmodel,vNmodel,vUmodel=None):
        #misfit statistic by each point. sum is the total misfit for the dataset.
        resid=self.calc_resid(vEmodel,vNmodel,vUmodel)
        point_misfit=np.zeros(np.size(self.vE))
        if self.is2d:
            dim=2
        else:
            dim=3
        for i in range(np.size(self.vE)):
            i0=dim*i
            i1=dim*(i+1)
            point_misfit[i]=resid[i0:i1].dot(self.inv_cov[i].dot(resid[i0:i1]))
        return point_misfit
        
    def calc_resid(self,vEmodel,vNmodel,vUmodel=None):
        if self.is2d or vUmodel is None:
            resid=np.ravel(np.column_stack((self.vE-vEmodel,self.vN-vNmodel)))
        else:
            resid=np.ravel(np.column_stack((self.vE-vEmodel,self.vN-vNmodel,self.vU-vUmodel)))
        return resid
        
    def calc_cov(self):
        self.cov=[]
        self.inv_cov=[]
        for i in range(np.size(self.sigE)):
            varE=self.sigE[i]**2
            varN=self.sigN[i]**2
            covEN=self.sigE[i]*self.sigN[i]*self.rhoEN[i]
            if self.is2d:
                covi=np.array([[varE,  covEN],
                               [covEN, varN]])
            else:
                varU=self.sigU[i]**2
                covi=np.array([[varE,  covEN, 0.],
                               [covEN, varN,  0.],
                               [0.,    0.,  varU]])
            self.cov.append(covi)
            self.inv_cov.append(scipy.linalg.inv(covi))
        #sparse arrays turn out to be both slower and more complicated...
        #self.cov=scipy.sparse.block_diag(ctemp).tocsc()
        #self.inv_cov=scipy.sparse.linalg.inv(self.cov)

