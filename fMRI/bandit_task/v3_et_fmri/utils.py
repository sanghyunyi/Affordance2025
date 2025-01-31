import sys, cv2, os, time
import imagezmq
from scipy import ndimage
import threading, queue

url = 'tcp://35.162.4.163:5555'#/actionclass'

sender = imagezmq.ImageSender(connect_to=url)
CAM_CODE = 1

#### MRI room specific setting ####
y = 512 #TODO change the hand location x, y accordingly
x = 806
ratio = 0.6 # How much FOV you need. Larger the ratio, the smaller the hand. 0.6 is the default

y_start = int(y-240)
y_end = int(y+240)

x_start = int(x-360)
x_end = int(x+360)


####################################


def crop_frame(frame, w_t, h_t):
    h = frame.shape[0]
    w = frame.shape[1]

    if w_t > w and h_t > h:
        frame = cv2.copyMakeBorder(
            frame,
            top=int((h_t-h)/2),
            bottom=int((h_t-h)/2),
            left=int((w_t-w)/2),
            right=int((w_t-w)/2),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
    elif w_t > w and h_t <= h:
        frame = cv2.copyMakeBorder(
            frame,
            top=0,
            bottom=0,
            left=int((w_t-w)/2),
            right=int((w_t-w)/2),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        frame = frame[int((h-h_t)/2):int((h+h_t)/2), :]
    elif w_t <= w and h_t > h:
        frame = cv2.copyMakeBorder(
            frame,
            top=int((h_t-h)/2),
            bottom=int((h_t-h)/2),
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        frame = frame[:, int((w-w_t)/2):int((w+w_t)/2)]
    else:
        frame = frame[int((h-h_t)/2):int((h+h_t)/2), int((w-w_t)/2):int((w+w_t)/2)]
    return frame

def for_video_capture_in_mri_room(img):
    img = img[y_start:y_end, x_start:x_end]
    #l = int((y_end - y_start)*2/3)
    #img = cv2.resize(img, (l, l), interpolation=cv2.INTER_AREA)
    #img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) # for 90 degree rotation, this is much faster
    img = crop_frame(img, 720, 480)
    cv2.waitKey(int(1000/60))
    return img

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(10, 10),
          font_scale=1,
          font_thickness=2,
          text_color=(0, 0, 0),
          text_color_bg=(255, 255, 255)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = 260, 20 #text_size
    cv2.rectangle(img, (0, 0), (x + text_w + 10, y + text_h + 10), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


class VideoCapture:
    def __init__(self, name):
        self.name = name
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.q_trial = []
        self.q_sess = []
        self.stop = False
        self.t = threading.Thread(target=self._reader, args=(lambda: self.stop, ))
        self.t.daemon = True
        self.t.start()
        print("Video capture thread start")

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self, stop):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print('disconnected')
                self.cap = cv2.VideoCapture(self.name)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.cap.set(cv2.CAP_PROP_FPS, 60)
                continue
            if len(self.q_trial) > 3:
                self.q_trial.pop()
            if len(self.q_sess) > 3:
                self.q_sess.pop()
            self.q_trial.append(frame)
            self.q_sess.append(frame)
            if stop():
                break

    def read(self, t):
        if t == 't':
            if len(self.q_trial) == 0:
                return False, None
            else:
                return True, self.q_trial.pop()
        elif t == 's':
            if len(self.q_sess) == 0:
                return False, None
            else:
                return True, self.q_sess.pop()

    def release(self):
        self.stop = True
        self.t.join()
        self.cap.release()

def captureVideo(stop, path, sessionClock, stream, typ):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoSave = cv2.VideoWriter(path +'.mov', fourcc, 60, (720, 480))
    ret, img = stream.read(typ)
    while True:
        ret, img = stream.read(typ)
        if ret :
            img = for_video_capture_in_mri_room(img)
            timeStamp = 'time:' + str(round(sessionClock.getTime(), 3))
            draw_text(img, timeStamp)
            videoSave.write(img)
        if stop():
            print("stopped")
            break
        #cv2.waitKey(int(1000/60))
    videoSave.release()
    return



######################################################################
######################################################################

def rescale_frame_(frame, percent=75): #deprecated
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def getResponse_(stream): #deprecated
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



