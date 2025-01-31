import sys, cv2, os, time
import imagezmq

url = 'tcp://52.42.34.109:5555'#/actionclass'

sender = imagezmq.ImageSender(connect_to=url)
CAM_CODE = 0

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def getResponse(stream): #deprecated
    #Opening OpenCV stream
    output = None
    process_duration = None
    prediction = None
    start_time = None
    end_time = None
    n = 0
    start_flag = False
    while True:
        ret, img = stream.read()
        img = rescale_frame(img, percent=10)
        prob = sender.send_image('HandPose', img)
        prob = prob.decode().split(",")
        prob = [float(p) for p in prob]
        print(prob)
        if start_flag == False:
            if sum(prob) < 1.-sum(prob):
                start_flag = True
                continue
            else:
                continue
        else: # if the subject showed palming at first
            new_prediction = str(prob.index(max(prob)) + 1)
            if max(prob) > 0.95:
                if n == 0:
                    start_time = time.time()
                    n += 1
                else:
                    if new_prediction == prediction:
                        n += 1
                    else:
                        n = 0
                prediction = new_prediction
            if n > 4:
                end_time = time.time()
                break
    process_duration = end_time - start_time
    output = prediction
    return output, process_duration



