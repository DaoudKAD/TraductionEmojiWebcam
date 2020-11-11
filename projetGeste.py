
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

# 0 -> F et 1 -> V

# Modele CN

class CNNGest(nn.Module):
    def __init__(self):
        super(CNNGest, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
i = 0

PATH = "modelGeste.pt"

# Load
model = torch.load(PATH)
chrono = time.time()

while( cap.isOpened() ):
	ret, frame = cap.read()
	if ret == True:
		
		if i == -1 : 
			frame = cv2.flip(frame,1)
			cv2.imshow('Traduction de gestes de la main en emoji' , frame)

		if i == 0 : # GESTE OK
			frame = cv2.flip(frame,1)
			img1 = frame
			img2 = cv2.imread('ok.png')
			img2 = cv2.resize(img2, (100,100))
			img3 = img1.copy()

			img3[10:110,250:350,:] = img2[0:100,0:100,:]
			cv2.imshow('Traduction de gestes de la main en emoji', img3)
			
		if i == 1 : # GESTE TWO
			frame = cv2.flip(frame,1)
			img1 = frame
			img2 = cv2.imread('two.png')
			img2 = cv2.resize(img2, (100,100))
			img3 = img1.copy()

			img3[10:110,250:350,:] = img2[0:100,0:100,:]
			cv2.imshow('Traduction de gestes de la main en emoji', img3)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


		if time.time() - chrono > 1 : 
			with torch.no_grad():
				scaler = StandardScaler()
				img = cv2.resize(frame, (28,28))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				output = torch.tensor(img).float()
				output = torch.tensor(scaler.fit_transform(output.clone().detach())).float()
				output = output.view(1,1,28,28)
				pred = model.forward(torch.tensor(output))
				choix = torch.argmax(pred)
				i = choix.item()

				chrono = time.time()

	else:
		break

cap.release()

cv2.destroyAllWindows()
