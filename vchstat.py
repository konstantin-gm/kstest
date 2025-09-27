#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 12:03:51 2023

@author: konstantin
"""
import random
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import allantools as allan
#import plotly.graph_objects as go


def covar_by_noise2d3d(q, dt):
  '''
  Ковариационная матрица дискретного случайного процесса
  Вызывается covar_by_noise2d либо covar_by_noise3d
  в зависимости от длины q
  
  '''  
  if len(q) == 2:
    Q = covar_by_noise2d(q, dt)
  elif len(q) == 3:
    Q = covar_by_noise3d(q, dt)
  else:
    Q = None
    
  return Q


def covar_by_noise2d(q, dt):
  '''
  Ковариационная матрица дискретного случайного процесса:
  белый частотный шум + случайные блуждания частоты
  
  Входы:
   q[0] = q1 - интенсивность белого частотного шума
   q[1] = q2 - "интенсивность" шума случайных блужданий частоты
   dt - интервал времени между отсчетами дискретного процесса

  Выход:
    Q - ковариационная матрица 2x2
  '''  
  Q = np.array([[q[0]*dt+q[1]*dt**3/3, q[1]*dt**2/2],
                   [q[1]*dt**2/2,         q[1]*dt]], dtype=np.double)
  return Q


def covar_by_noise3d(q, dt):
  '''
  Ковариационная матрица дискретного случайного процесса:
  белый частотный шум + случайные блуждания частоты + сл. блужд. изм. частоты
  
  Входы:
   q[0] = q1 - интенсивность белого частотного шума
   q[1] = q2 - "интенсивность" шума случайных блужданий частоты
   q[2] = q3 - "интенсивность" шума случайных блужданий изменения частоты
   dt - интервал времени между отсчетами дискретного процесса

  Выход:
    Q - ковариационная матрица 2x2
  '''  
  Q = np.array([[q[0]*dt+q[1]*dt**3/3+q[2]*dt**5/20, q[1]*dt**2/2+q[2]*dt**4/8, q[2]*dt**3/6],
                [q[1]*dt**2/2+q[2]*dt**4/8, q[1]*dt+q[2]*dt**3/3, q[2]*dt**2/2],
                [q[2]*dt**3/6, q[2]*dt**2/2, q[2]*dt]], dtype=np.double)
  
  return Q


def q0_by_wpnadev(tau, adev):
  '''
  Входы:
    tau - интервал времени измерения, в секундах
    adev - значение СКДО

  Выход:
    (q0) - интенсивность белого фазового шума
  '''
  return (adev*tau)**2. / 3.


def q1_by_wfnadev(tau, adev):
  '''
  Входы:
    tau - интервал времени измерения, в секундах
    adev - значение СКДО

  Выход:
    (q1) - интенсивность белого частотного шума
  '''
  return adev**2 * tau


def q2_by_rwfnadev(tau, adev):
  '''
  Входы:
    tau - интервал времени измерения, в секундах
    adev - значение СКДО

  Выход:
    (q2) - интенсивность шума случайных блужданий частоты
  '''
  return adev**2. / tau * 3


def generate_tau():
    
  tau = np.arange(1,10)
  tau = np.append(tau, np.arange(10,100,10))
  tau = np.append(tau, np.arange(100,1000,100))
  tau = np.append(tau, np.arange(1000,10000,1000))
  tau = np.append(tau, np.arange(10000,100000,10000))
  tau = np.append(tau, np.arange(100000,1000000,100000))
  tau = np.append(tau, np.arange(1000000,10000000,1000000))
  
  return tau


def allan_deviation(z, dt, tau, min_ave=3):
  '''
  Девиация Аллана (СКДО)

  Входы:
    z - отсчеты фазы, в секундах
    dt - интервал времени между отсчетами, в секундах
    tau - интервалы времени измерения, для которых нужно рассчитать СКДО,
    в единицах dt

  Выходы:
    tau[:maxi].astype(np.double)*dt - массив значений tau, для которых удалось 
    рассчитать СКДО, в секундах
    ADEV[:maxi] - значения СКДО
  '''
  ADEV = np.zeros(tau.size, dtype='double')
  n = z.size
  maxi = 0
  for i in range(tau.size):
    if tau[i]*(min_ave+2) < n:
        maxi = i
        sigma2 = np.sum((z[2*tau[i]::1] - 2*z[tau[i]:-tau[i]:1] + z[0:-2*tau[i]:1])**2)
        ADEV[i] = np.sqrt(0.5*sigma2/(n-2*tau[i]))/tau[i]/dt
    else:
        break  
  return tau[:maxi].astype(np.double)*dt, ADEV[:maxi]


def parabolic_deviation(z, dt, tau):
  '''
  Параболическая девиация

  Входы:
    z - отсчеты фазы, в секундах
    dt - интервал времени между отсчетами, в секундах
    tau - интервалы времени измерения, для которых нужно рассчитать СКДО,
    в единицах dt

  Выходы:
    tau[:maxi].astype(np.double)*dt - массив значений tau, для которых удалось 
    рассчитать PDEV, в секундах
    PDEV[:maxi] - значения параболической девиации
  '''
  PDEV = np.zeros(tau.size, dtype='double')
  n = z.size
  maxi = 0
  for i in range(tau.size):
    if tau[i]*3 < n:
      maxi = i
      M = 0
      Si = 0
      c1 = np.polyfit(range(0,tau[i]+1,1),z[0:tau[i]+1:1],1)
      for j in range(tau[i],n-tau[i],tau[i]):
        c2 = np.polyfit(range(j,j+tau[i]+1,1),z[j:j+tau[i]+1:1],1)
        Si += (c1[0] - c2[0])**2
        M += 1
        c1 = c2
        PDEV[i] = np.sqrt(0.5*Si/M)/dt
    else:
        break    
    return tau[:maxi].astype(np.double)*dt, PDEV[:maxi]


def getF2d(dt):
  F = np.array([[1, dt], [0, 1]], dtype=np.double)  
  return F


def getF3d(dt):
  F = np.array([[1, dt, dt**2/2], [0, 1, dt], [0, 0, 1]], dtype=np.double)
  return F


class Model2d3d:

  def __init__(self, noise, dt, phase0, freq0, drift):
    '''
    noise - [q0, q1, q2] интенсивности шумов (белый фазовый, белый частотный, 
    шум случайных блужданий частоты)
    dt - интервал времени, в секундах
    freq - отклонение частоты, в отн. ед.
    drift - суточный дрейф частоты, в отн. ед.
    '''
    
    nd = len(noise)
    self.nd = nd
    
    if nd == 4:
      self.F = getF3d(dt)
      self.D = np.array([0.5*dt**2, dt, 0])
    else:
      self.F = getF2d(dt)
      self.D = np.array([0.5*dt**2, dt])
      
    self.q = noise
    if nd == 1:
      self.Q = 1
      self.L = 1
    else:
      self.Q = covar_by_noise2d3d(self.q[1:], dt)
      self.L = linalg.cholesky(self.Q)    
    
    self.dt = dt    
        
    self.X = np.zeros(2)
    self.X[0] = phase0
    self.X[1] = freq0
    self.drift = drift    
  
  def generate(self, N, NofCtrls=0):
    '''
    Вход:
      N - количество отчетов
    Выход:
      phase - отчсчеты фазы numpy.array(N)
    '''
    
    self.phase = np.zeros(N)
    sq0 = np.sqrt(self.q[0])
    
    controls = random.sample(range(1, N), NofCtrls)
    
    if self.nd == 1:
      for i in range(N):
        self.X = self.F@self.X + self.D*self.drift
        wpn = np.random.randn(1)*sq0
        self.phase[i] = self.X[0] + wpn[0]
    else:
      for i in range(N):
        w = np.random.randn(self.nd-1)
        self.X = self.F@self.X + self.L@w + self.D*self.drift
        if i in controls:
            self.X[1] += random.uniform(-1e-11, 1e-11)
        wpn = np.random.randn(1)*sq0
        self.phase[i] = self.X[0] + wpn[0]

    return self.phase, self.X[1]


class Kalman:
  def __init__(self, F, H, Q, R):
    self.F = F
    self.H = H
    self.Q = Q
    self.R = R
  
  def filter(self, x0, P0, z):
    x = x0
    P = P0    
    x_filtered = []
    y_filtered = []
    k_list = []
    K = np.ones((2, 1))
    n = len(x0)
    for zk in z:
      x = self.F @ x
      P = self.F @ P @ self.F.T + self.Q

      if K[1] >= 1.e-5:
        K = P @ self.H.T / (self.H @ P @ self.H.T + self.R)      
      P = (np.eye(n) - np.outer(K, self.H))@P

      x += K*(zk - self.H @ x)

      x_filtered.append(x[0][0])
      y_filtered.append(x[1][0])
      k_list.append(K)
    
    return x_filtered, y_filtered, k_list


def exp_filter(x, a):
    n = len(x)
    xfilt = np.zeros(n)
    xfilt[0] = x[0]
    for i in range(n-1):
        xfilt[i+1] = a*x[i+1] + (1-a)*xfilt[i]
    return xfilt


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
