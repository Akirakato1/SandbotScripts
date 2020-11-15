#tensorflow installation 
#https://www.youtube.com/watch?v=usR2LQuxhL4

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

import ops as utils_ops
import label_map_util
import visualization_utils as vis_util

from Detection_Model import Detection_Model

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

class PathCalculator: 
    #https://www.rapidtables.com/web/tools/pixel-ruler.html
    #https://mycurvefit.com/
    
    def __init__(self, image_np,image_npO):
        self.compressResolution=(50,50)
        self.image_np=np.array(Image.fromarray(image_np.copy())
                               .resize(self.compressResolution,Image.ANTIALIAS))
        self.image_npO=np.array(Image.fromarray(image_npO.copy())
                                .resize(self.compressResolution,Image.ANTIALIAS))
        self.solutionPPath=list()
        self.distancePenalty=1.5
        self.width=self.image_np.shape[1]
        self.height=self.image_np.shape[0]
        self.scalingFactor=(self.width/self.compressResolution[0]*20,self.height/self.compressResolution[1]*2000)
        self.startingPixel=(self.height-2,17)
        #self.startingPixel=(self.height-2,self.width//2)
        self.endingPixel=(1,self.width//2)
        print(self.startingPixel)
        print(self.endingPixel)
        self.isBorder={}
        self.pixelToNode={}
        self.NodeToPixel={}
        count=0
        newImage=self.image_np.copy()
        for i in range(self.height):
            for j in range(self.width):
                self.pixelToNode[(i,j)]=count
                self.NodeToPixel[count]=(i,j)
                pixel=self.image_np[i][j]
                if pixel[0]==255 and pixel[1]==255 and pixel[2]==255:
                    self.isBorder[count]=False
                else:
                    self.isBorder[count]=True
                    newImage[i][j][0]=0
                    newImage[i][j][1]=0
                    newImage[i][j][2]=0
                count+=1
        display(Image.fromarray(self.image_np))
        display(Image.fromarray(newImage))
        #print([str(self.NodeToPixel[key])+":"+str(self.isBorder[key]) for key in self.isBorder.keys() if self.isBorder[key]==True])
        self.startingNode=self.pixelToNode[self.startingPixel]
        self.endingNode=self.pixelToNode[self.endingPixel]
        print(self.startingNode)
        print(self.endingNode)
        self.graph=list()
        for i in range(self.height):
            for j in range(self.width):
                '''
                if i==0 or j==0 or i==self.height-1 or j==self.width-1:
                    adj=[1 for i in range(self.width*self.height)]
                    currentNode=self.pixelToNode[(i,j)]
                    self.graph.append(adj)
                else:
                '''
                adj=[0 for i in range(self.width*self.height)]
                currentNode=self.pixelToNode[(i,j)]
                if i>0:
                    uNode=self.pixelToNode[(i-1,j)]
                    if self.isBorder[uNode]:
                        adj[uNode]=9999
                    else:
                        adj[uNode]=abs(j-self.startingPixel[1])*self.distancePenalty+1

                if j>0:
                    lNode=self.pixelToNode[(i,j-1)]
                    if self.isBorder[lNode]:
                        adj[lNode]=9999
                    else:
                        adj[lNode]=abs(j-1-self.startingPixel[1])*self.distancePenalty+1

                if j<self.width-1:
                    rNode=self.pixelToNode[(i,j+1)]
                    if self.isBorder[rNode]:
                        adj[rNode]=9999
                    else:
                        adj[rNode]=abs(j+1-self.startingPixel[1])*self.distancePenalty+1

                if i<self.height-1:
                    dNode=self.pixelToNode[(i+1,j)]
                    if self.isBorder[dNode]:
                        adj[dNode]=9999
                    else:
                        adj[dNode]=abs(j-self.startingPixel[1])*self.distancePenalty+1
                
                #print(currentNode)
                #print(adj)
                self.graph.append(adj)
    
    def pixelCoveredToDistanceCoveredRL(self,pixelStart, pixelEnd):
        #y = 1234.47 + (0.004702566 - 1234.47)/(1 + (x/8289.626)^3.058906) 
        #pixel away from bottom->distance(ft) covered per pixel left to right
        
        pixelAway=self.height-pixelStart[0]
        pixelCovered=abs(pixelStart[1]-pixelEnd[1])
        
        return (1234.47 + (0.004702566 - 1234.47)/(1 + (pixelAway/8289.626)**3.058906))*pixelCovered*self.scalingFactor[0]
        #return (6092.447 + (0.005479013 - 6092.447)/(1 + (pixelAway/215.6)**5.382167))*pixelCovered
        #return y = -17854660 + (325.4808 - -17854660)/(1 + (x/59954040)^0.7429366)
        
    def pixelCoveredToDistanceCoveredFB(self,pixelStart, pixelEnd):
        #y = 2192701 + (1.927978 - 2192701)/(1 + (x/9308.816)^3.33469)
        #pixel away from bottom->total distance(ft) covered front to back
        
        pixelAwayStart=self.height-pixelStart[0]
        pixelAwayEnd=self.height-pixelEnd[0]
        
        d1=2192701 + (1.927978 - 2192701)/(1 + (pixelAwayStart/9308.816)**3.33469)*self.scalingFactor[1]
        d2=2192701 + (1.927978 - 2192701)/(1 + (pixelAwayEnd/9308.816)**3.33469)*self.scalingFactor[1]
        #d1=2651658 + (2.27663 - 2651658)/(1 + (pixelAwayStart/139.5231)**6.530391)
        #d2=2651658 + (2.27663 - 2651658)/(1 + (pixelAwayEnd/139.5231)**6.530391)
        return (d2-d1)
    
    def minDistance(self,dist,queue): 
        # Initialize min value and min_index as -1 
        minimum = float("Inf") 
        min_index = -1
          
        for i in range(len(dist)): 
            if dist[i] < minimum and i in queue: 
                minimum = dist[i] 
                min_index = i 
        return min_index 
  
    def getPath(self, parent, j): 
        output=list()
        #Base Case : If j is source 
        if parent[j] == -1 :  
            #print(j),
            output.append(j)
            return output
        output=self.getPath(parent , parent[j]) 
        #print(j), 
        output.append(j)
        return output
          
    def getSolution(self, dist, parent): 
        src = self.startingNode
        #print("Vertex \t\tDistance from Source\tPath") 
        solutions=list()
        for i in range(1, len(dist)): 
            #print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src, i, dist[i])), 
            solutions.append(self.getPath(parent,i))
        return solutions
    
    '''Function that implements Dijkstra's single source shortest path 
    algorithm for a graph represented using adjacency matrix 
    representation'''
    def dijkstra(self): 
  
        row = len(self.graph) 
        col = len(self.graph[0]) 
  
        # Initialize all distances as INFINITE  
        dist = [float("Inf")] * row 
  
        parent = [-1] * row 
  
        dist[self.startingNode] = 0

        queue = [] 
        for i in range(row): 
            queue.append(i) 
              
        #Find shortest path for all vertices 
        while queue: 
  
            u = self.minDistance(dist,queue)
            queue.remove(u) 
            
            for i in range(col): 
                '''Update dist[i] only if it is in queue, there is 
                an edge from u to i, and total weight of path from 
                src to i through u is smaller than current value of 
                dist[i]'''
                if self.graph[u][i] and i in queue: 
                    if dist[u] + self.graph[u][i] < dist[i]: 
                        dist[i] = dist[u] + self.graph[u][i] 
                        parent[i] = u 
        self.solutionPPath=[self.NodeToPixel[n] for n in self.getSolution(dist,parent)[self.endingNode-1]]
        print(self.solutionPPath)
        return self.solutionPPath
    
    def addPathToImage(self):
        for i in range(self.height):
            for j in range(self.width):
                if (i,j) in self.solutionPPath:
                    self.image_np[i][j][0]=255
                    self.image_np[i][j][1]=0
                    self.image_np[i][j][2]=0
                    self.image_npO[i][j][0]=255
                    self.image_npO[i][j][1]=0
                    self.image_npO[i][j][2]=0
    
    def pixelInstructions(self):
        if len(self.solutionPPath)==0:
            print("call dijkstra first to calculate path")
            return;
        self.addPathToImage()
        display(Image.fromarray(self.image_np).resize((300,300),Image.ANTIALIAS))
        display(Image.fromarray(self.image_npO).resize((300,300),Image.ANTIALIAS))
        
        directionP=list()
        instructionP=list()
        
        currentP=self.solutionPPath[0]
        for p in self.solutionPPath[1:len(self.solutionPPath)]:
            directionP.append(self.getDirection(currentP,p))
            currentP=p
        
        currentDirection=directionP[0]
        count=1
        for i in range(1,len(directionP)):
            if (not directionP[i]==currentDirection) or i==len(directionP)-1:
                instructionP.append((currentDirection,count))
                currentDirection=directionP[i]
                count=1
            else:
                count+=1
        
        print(directionP)
        print(instructionP)
        return instructionP
    
    #distance is in feet
    def getDistanceInstruction(self):
        pI=self.pixelInstructions()
        distanceI=list()
        pixelCount=0
        for I in pI:
            if I[0]=="forward":
                startPixel=self.solutionPPath[pixelCount]
                pixelCount+=I[1]
                endPixel=self.solutionPPath[pixelCount]
                distanceI.append(("forward",self.pixelCoveredToDistanceCoveredFB(startPixel,endPixel)))
                
            elif I[0]=="backward":
                startPixel=self.solutionPPath[pixelCount]
                pixelCount+=I[1]
                endPixel=self.solutionPPath[pixelCount]
                distanceI.append(("backward",self.pixelCoveredToDistanceCoveredFB(startPixel,endPixel)))
                
            elif I[0]=="left":
                startPixel=self.solutionPPath[pixelCount]
                pixelCount+=I[1]
                endPixel=self.solutionPPath[pixelCount]
                distanceI.append(("left",self.pixelCoveredToDistanceCoveredRL(startPixel,endPixel)))
                
            elif I[0]=="right":
                startPixel=self.solutionPPath[pixelCount]
                pixelCount+=I[1]
                endPixel=self.solutionPPath[pixelCount]
                distanceI.append(("right",self.pixelCoveredToDistanceCoveredRL(startPixel,endPixel)))
                
            else:
                print("error something gone wrong in distance isntruction")
        
        print(distanceI)
        return distanceI
            
    #p1 is current pixel, p2 is the next pixel
    def getDirection(self,p1,p2):
        if p2[0]<p1[0]:
            direction="forward"
        elif p2[0]>p1[0]:
            direction="backward"
        elif p2[1]<p1[1]:
            direction="left"
        elif p2[1]>p1[1]:
            direction="right"
        else:
            direction="error"
        return direction
    
    
class PathPredictor:
    
    def __init__(self):
        self.dm=Detection_Model(model_name)
        
    def predictPath(self,imgpath):
        imageTuple=self.dm.show_inference(imgpath)
        g=PathCalculator(imageTuple[0],imageTuple[1])
        g.dijkstra()
        dI=g.getDistanceInstruction()
        return dI