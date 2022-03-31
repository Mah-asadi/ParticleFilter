# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 13:30:39 2022

@author: Mahboubeh
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
from numpy.random import randn
import scipy
from numpy.linalg import norm
import scipy.stats
from filterpy.monte_carlo import systematic_resample
import pygame
from pygame.locals import *
import sys

def uniform_particles(x_range, y_range, theta_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(theta_range[0], theta_range[1], size=N)
    return particles

def predict(particles, ww, ws):
    N = len(particles)
    noise_x = np.random.normal(0, ws, N)
    noise_y = np.random.normal(0, ws, N)
    noise_theta = np.random.normal(0, ww, N)
    particles[:, 2] += us/L + noise_theta
    particles[:, 0] += uw * np.cos(particles[:, 2])  + noise_x
    particles[:, 1] += uw * np.sin(particles[:, 2])  + noise_y
    
def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])
        
    weights += 1.e-300      
    weights /= sum(weights)

def estimate(particles, weights):
    pos = particles[:, 0:2]
    mean = np.average(pos, weights = weights, axis=0)
    var  = np.average((pos - mean)**2, weights = weights, axis=0)
    return mean, var

def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.
    indexes = np.searchsorted(cumulative_sum, np.random(N))
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))

N = 1000
iter = 600
L = 0.20
landmarks = np.array([[10, 10, 0]])
NL = len(landmarks)
particles = uniform_particles((0, 20), (0, 20), (0, 6.28), N)
weights = np.ones(N)/N
robot = np.array([15 ,10, 3.14/2])
ur = 0.5341/2
ul = 0.5131/2
us = ur-ul
uw = (ur + ul)/2

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (0, 255, 0)
BLUE = (50, 50, 255)
PURPLE = (255,0,255)
#######################
pygame.init()
SCREEN_SIZE = (700, 700)
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Particle Filter - 2D Robot')
pygame.display.flip()
screen.fill(WHITE)
scale = 35
fps = pygame.time.Clock()
paused = False
j = 0
#############################
while True:
    j += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                paused = not paused
    if j<iter: 
        zs = np.sqrt( (robot[0]-landmarks[0,0])**2 + (robot[1]-landmarks[0,1])**2) + np.random.normal(0, 0.1)

    # Ground Truth
    
        robot[2] += us/L
        robot[0] += uw * np.cos(robot[2]) 
        robot[1] += uw * np.sin(robot[2]) 

    
    # move particles to the next position acording to the control signal
        predict(particles, 0.1 , 0.01)
    
    # incorporate measurements
        z = np.array([zs])
        landmarks[0, 2] = robot[2]
        if j%8==0:
            update(particles, weights, z, 0.1, landmarks)

    # resample if too few effective particles
        NE = 1. / np.sum(np.square(weights))
        if NE<N/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1/N)
        mu, var = estimate(particles, weights)
        pygame.draw.circle(screen, BLACK, (350,350), 5*scale, width=1)
        pygame.draw.circle(screen, RED, (350,350), 5)
        for i in range (len(particles)):
            pygame.draw.circle(screen, BLUE, (particles[i, 0]*scale, particles[i, 1]*scale), 5, width=2)
        pygame.draw.circle(screen, PURPLE, (mu[0]*scale, mu[1]*scale), 5)
        pygame.draw.circle(screen, GREEN, (robot[0]*scale, robot[1]*scale), 5)
        pygame.display.update()
        fps.tick(10)
        screen.fill(WHITE)
    
    pygame.display.update()