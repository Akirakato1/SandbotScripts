import numpy as np
import PIL as pil
from PathPredictor import PathPredictor

class SandBot:
    
    def __init__(self, botId, initGPS, initIMU):
        self.id=botId
        self.gps=[initGPS]
        self.imu=[initIMU]
        self.pPath=list()
        
    def predictPath(self, imgpath, pp):
        self.pPath=pp.predictPath(imgpath)
        
    def updateGPSIMU(self, GPS, IMU):
        self.gps.append(GPS)
        self.imu.append(IMU)
    
    def getGPSHistory(self):
        return self.gps
    
    def getId(self):
        return self.id
        
class Model:
    
    def __init__(self):
        self.units=list()
        self.idCount=0
        self.pathPredictor=PathPredictor()
    
    def addUnit(self, initGPS, initIMU):
        self.units.append(SandBot(self.idCount,initGPS, initIMU))
        self.idCount+=1
        
    def predictUnit(self,unitId,img):
        for u in self.units:
            if u.getId()==unitId:
                u.
    
    
    