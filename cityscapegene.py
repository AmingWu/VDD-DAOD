import cv2
import numpy as np
import pdb

path1 = '/home/wam/newdecomposition/SW_Faster_ICR_CCR2/Results/'

tbike = path1 + 'comp4_det_test_bike.txt'
tbus = path1 + 'comp4_det_test_bus.txt'
tcar = path1 + 'comp4_det_test_car.txt'
tmotor = path1 + 'comp4_det_test_motor.txt'
tperson = path1 + 'comp4_det_test_person.txt'
trider = path1 + 'comp4_det_test_rider.txt'
ttruck = path1 + 'comp4_det_test_truck.txt'

bike = open(tbike, 'r')
bike = bike.readlines()
bus = open(tbus, 'r')
bus = bus.readlines()
car = open(tcar, 'r')
car = car.readlines()
motor = open(tmotor, 'r')
motor = motor.readlines()
person = open(tperson, 'r')
person = person.readlines()
rider = open(trider, 'r')
rider = rider.readlines()
truck = open(ttruck, 'r')
truck = truck.readlines()

## bicycle
for i in range(len(bike)):
    data = bike[i]
    pos = []
    for j in range(len(data)):
        if data[j] == ' ':
            pos.append(j)
    imgpath = str(data[0:pos[0]])
    img = cv2.imread(imgpath)
    score = float(data[pos[0]+1:pos[1]])

    if score > 0.5:
        print(i)
        ux = int(float(data[pos[1]+1:pos[2]]) + 0.5)
        uy = int(float(data[pos[2]+1:pos[3]]) + 0.5)
        dx = int(float(data[pos[3]+1:pos[4]]) + 0.5)
        dy = int(float(data[pos[4]+1:-1]) + 0.5)
        cv2.rectangle(img, (ux,uy), (dx,dy), (0,0,255), 2)
        cv2.imwrite(imgpath, img)

print('bicycle good')

## bus
for i in range(len(bus)):
    data = bus[i]
    pos = []
    for j in range(len(data)):
        if data[j] == ' ':
            pos.append(j)
    imgpath = str(data[0:pos[0]])
    img = cv2.imread(imgpath)
    score = float(data[pos[0]+1:pos[1]])

    if score > 0.5:
        print(i)
        ux = int(float(data[pos[1]+1:pos[2]]) + 0.5)
        uy = int(float(data[pos[2]+1:pos[3]]) + 0.5)
        dx = int(float(data[pos[3]+1:pos[4]]) + 0.5)
        dy = int(float(data[pos[4]+1:-1]) + 0.5)
        cv2.rectangle(img, (ux,uy), (dx,dy), (128,128,0), 2)
        cv2.imwrite(imgpath, img)

print('bus good')

## car
for i in range(len(car)):
    data = car[i]
    pos = []
    for j in range(len(data)):
        if data[j] == ' ':
            pos.append(j)
    imgpath = str(data[0:pos[0]])
    img = cv2.imread(imgpath)
    score = float(data[pos[0]+1:pos[1]])

    if score > 0.5:
        print(i)
        ux = int(float(data[pos[1]+1:pos[2]]) + 0.5)
        uy = int(float(data[pos[2]+1:pos[3]]) + 0.5)
        dx = int(float(data[pos[3]+1:pos[4]]) + 0.5)
        dy = int(float(data[pos[4]+1:-1]) + 0.5)
        cv2.rectangle(img, (ux,uy), (dx,dy), (255,0,0), 2)
        cv2.imwrite(imgpath, img)

print('car good')

## mbike
for i in range(len(motor)):
    data = motor[i]
    pos = []
    for j in range(len(data)):
        if data[j] == ' ':
            pos.append(j)
    imgpath = str(data[0:pos[0]])
    img = cv2.imread(imgpath)
    score = float(data[pos[0]+1:pos[1]])

    if score > 0.5:
        print(i)
        ux = int(float(data[pos[1]+1:pos[2]]) + 0.5)
        uy = int(float(data[pos[2]+1:pos[3]]) + 0.5)
        dx = int(float(data[pos[3]+1:pos[4]]) + 0.5)
        dy = int(float(data[pos[4]+1:-1]) + 0.5)
        cv2.rectangle(img, (ux,uy), (dx,dy), (128,128,64), 2)
        cv2.imwrite(imgpath, img)

print('mbike good')

## rider
for i in range(len(rider)):
    data = rider[i]
    pos = []
    for j in range(len(data)):
        if data[j] == ' ':
            pos.append(j)
    imgpath = str(data[0:pos[0]])
    img = cv2.imread(imgpath)
    score = float(data[pos[0]+1:pos[1]])

    if score > 0.5:
        print(i)
        ux = int(float(data[pos[1]+1:pos[2]]) + 0.5)
        uy = int(float(data[pos[2]+1:pos[3]]) + 0.5)
        dx = int(float(data[pos[3]+1:pos[4]]) + 0.5)
        dy = int(float(data[pos[4]+1:-1]) + 0.5)
        cv2.rectangle(img, (ux,uy), (dx,dy), (128,128,192), 2)
        cv2.imwrite(imgpath, img)

print('rider good')

## person
for i in range(len(person)):
    data = person[i]
    pos = []
    for j in range(len(data)):
        if data[j] == ' ':
            pos.append(j)
    imgpath = str(data[0:pos[0]])
    img = cv2.imread(imgpath)
    score = float(data[pos[0]+1:pos[1]])

    if score > 0.5:
        print(i)
        ux = int(float(data[pos[1]+1:pos[2]]) + 0.5)
        uy = int(float(data[pos[2]+1:pos[3]]) + 0.5)
        dx = int(float(data[pos[3]+1:pos[4]]) + 0.5)
        dy = int(float(data[pos[4]+1:-1]) + 0.5)
        cv2.rectangle(img, (ux,uy), (dx,dy), (180,105,255), 2)
        cv2.imwrite(imgpath, img)

print('person good')

## truck
for i in range(len(truck)):
    data = truck[i]
    pos = []
    for j in range(len(data)):
        if data[j] == ' ':
            pos.append(j)
    imgpath = str(data[0:pos[0]])
    img = cv2.imread(imgpath)
    score = float(data[pos[0]+1:pos[1]])

    if score > 0.5:
        print(i)
        ux = int(float(data[pos[1]+1:pos[2]]) + 0.5)
        uy = int(float(data[pos[2]+1:pos[3]]) + 0.5)
        dx = int(float(data[pos[3]+1:pos[4]]) + 0.5)
        dy = int(float(data[pos[4]+1:-1]) + 0.5)
        cv2.rectangle(img, (ux,uy), (dx,dy), (0,0,64), 2)
        cv2.imwrite(imgpath, img)

print('truck good')