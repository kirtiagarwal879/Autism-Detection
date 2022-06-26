import csv
import cv2
import socket, os
from gaze_tracking import GazeTracking
#Basically,this program is our base code from where we import all other program files as well as we call our function
#and from this file we predict our result, we take the video input from this file and store our manual and predicted gaze in the csv file and print the output on console screen



gaze = GazeTracking()
cam = cv2.VideoCapture('Data/r1.mp4')
row  = [["FrameNo", "PredictGaze", "ManualGaze"]]
dict = [0,0,0,0,0]
frameno = 0
while True:
    _, frame = cam.read()
    if frame is None:
        break
    

    gaze.refresh(frame)
    if(frameno > 1700):
        break
    frame = gaze.annotated_frame()
    if(frameno % 2 == 0):
        text = 0
        STR = ""
        if gaze.is_blinking():
            text = 1 
            STR =  "EYE BLINKING"
        elif gaze.is_right():
            text = 2 
            STR =  "LOOKING RIGHT"
        elif gaze.is_left():
            text = 3
            STR =  "LOOKING LEFT"
        elif gaze.is_center():
            text = 2 
            STR = "LOOKING RIGHT"

        cv2.putText(frame, STR, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        cv2.imshow("Demo", frame)

        tmp = [int(frameno/2),text,text]
        #print("Frame No",tmp[0])
        dict[text] += 1
        row.append(tmp)

        if cv2.waitKey(1) == 27:
            break
    frameno += 1


#report = open('outputofautism.txt','w'); 

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~OUTPUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Blinking(Horizontal_Eye_Length/Vertical_Eye_Length)= ", dict[0])
print("Right Direction= " , dict[1] + dict[3])
print("Left Direction= ", dict[2])
print(dict)

print("~~~~~~~~~~~~~~~~~~~~ Autism Spectrum Detection Or Not~~~~~~~~~~~~~~~~~~~~~~~~~~~")
if(dict[1] > dict[2]):
    print("This child is diagnosed as Autistic ")
    print( "Average gaze ratio per second  = ", gaze.accuracy());
else:
    print("This child is not diagnosed as Autistic")
    print( "Average gaze ratio per second  = ", gaze.Accuracy());


with open('CSV_Data/4.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row)
