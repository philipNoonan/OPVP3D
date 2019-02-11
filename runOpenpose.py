import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import pyopenpose as op

cap = cv2.VideoCapture(0)

parser = argparse.ArgumentParser()
opWrapper = op.WrapperPython()

params = dict()
params["model_folder"] = "D://models//"

opWrapper.configure(params)

opWrapper.start()

while(True):
    datum = op.Datum()
    # Capture frame-by-frame
    ret, imageToProcess = cap.read()
    datum.cvInputData = imageToProcess
	opWrapper.emplaceAndPop([datum])
	
    # Display the resulting frame
    cv2.imshow('frame',datum.cvOutputData)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()