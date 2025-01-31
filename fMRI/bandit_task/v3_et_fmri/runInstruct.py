from psychopy import visual, event, core
import numpy as np
import os, cv2
from config import *
from utils import *

def runInstruct(expInfo, dispInfo, taskInfo, taskObj, keyInfo, instruct_initInfo):
    pages = np.arange(instruct_initInfo.instructInfo.numPages)
    # Initialize instruct page object
    instructObj = InstructPages(taskObj, dispInfo, instruct_initInfo.instructInfo, instruct_initInfo.imagePaths)

    stream = VideoCapture(CAM_CODE)
    while True:
        ret, img = stream.read('r')
        if ret:
            break
    img = rescale_frame(img)
    _ = sender.send_image('HandPose', img)

    sessionClock = core.Clock()
    # TODO
    # Start from the hand gesture excercise
    currPage = 10
    while True:
        # Run instruction function

        getattr(instructObj, 'instruct_%s' % currPage)()
        # wait for response
        if currPage >= 12 and currPage <=17: # interactive page
            print('Page: ' + str(currPage))
            response = None
            start_flag = False
            count = 0
            prediction = None
            while True:
                # process hand gesture response
                # map 1: pinch, 2: clench, 3: poke
                ret, img = stream.read('r')
                if ret:
                    img = rescale_frame(img)
                    prob = sender.send_image('HandPose', img)
                    prob = prob.decode().split(",")
                    prob = [float(p) for p in prob]
                    print("pinch: ", prob[0], "clench: ", prob[1], "poke: ", prob[2])
                    if start_flag == False:
                        if max(prob) < 1.-sum(prob):
                            start_flag = True
                            print('Palm recognized')
                            continue
                        else:
                            continue
                    else:
                        new_prediction = str(prob.index(max(prob)) + 1)
                        if max(prob) > 0.95:
                            if new_prediction == prediction:
                                count += 1
                            else:
                                hand_start_time = sessionClock.getTime()
                                prediction = new_prediction
                                count = 0
                        if count >= 1:
                            hand_end_time = sessionClock.getTime()
                            process_duration = hand_end_time - hand_start_time
                            if process_duration > 0.5:
                                response = prediction
                                break
                if not ret:
                    print("Missed frame!")
                cv2.waitKey(int(1000/30))

            # Process hand gesture response
            # Which response was made
            if keyInfo.resp1 in response:
                # Pinch gesture was made
                text = visual.TextStim(win=instructObj.screen,
                        text="You Pinched. \n\n Make the palm position and be ready for the next trial.",
                        pos=instructObj.posMid,
                        height=instructObj.height,
                        color='red',
                        wrapWidth=instructObj.wrapWidth)

                text.draw()
                instructObj.screen.flip()
                core.wait(2)
                if currPage == 12 or currPage == 15:
                    currPage += 1
                else:
                    continue

            elif keyInfo.resp2 in response:
                # Clench gesture was made
                text = visual.TextStim(win=instructObj.screen,
                        text="You Clenched. \n\n Make the palm position and be ready for the next trial.",
                        pos=instructObj.posMid,
                        height=instructObj.height,
                        color='red',
                        wrapWidth=instructObj.wrapWidth)

                text.draw()
                instructObj.screen.flip()
                core.wait(2)
                if currPage == 13 or currPage == 16:
                    currPage += 1
                else:
                    continue

            elif keyInfo.resp3 in response:
                # Poke gesture was made
                text = visual.TextStim(win=instructObj.screen,
                        text="You Poked. \n\n Make the palm position and be ready for the next trial.",
                        pos=instructObj.posMid,
                        height=instructObj.height,
                        color='red',
                        wrapWidth=instructObj.wrapWidth)

                text.draw()
                instructObj.screen.flip()
                core.wait(2)
                if currPage == 14 or currPage == 17:
                    currPage += 1
                else:
                    continue

        elif currPage == instruct_initInfo.instructInfo.numPages:
            response = event.waitKeys(keyList=[keyInfo.instructDone, 'escape'])
            if keyInfo.instructDone in response:
                break
            elif 'escape' in response:
                core.wait(1)
                core.quit()
        else:
            response = event.waitKeys(keyList=keyInfo.instructAllow)
            if keyInfo.instructPrev in response:
                # Move back a screen
                if currPage == 18:
                    currPage = max(10,currPage-7)
                    print('Page: '  + str(currPage))
                else:
                    currPage = max(10,currPage-1)
                    print('Page: '  + str(currPage))
            elif keyInfo.instructNext in response:
                # Move forward a screen
                currPage = min(len(pages), currPage+1)
                print('Page: ' + str(currPage))
            elif 'escape' in response:
                core.wait(1)
                core.quit()
    return



def initInstruct(expInfo, taskInfo, taskObj):
    # Specify instruction image directory
    instructDir = expInfo.stimDir + os.sep + 'instruct'

    def setImagePath(instructDir):
        # Instruction related images
        resp_pic = instructDir + os.sep + 'resp_pic.png' # apparatus
        object_img = instructDir + os.sep + 'object_img.png'
        leftStim_path = instructDir + os.sep + 'pinch.mp4'
        midStim_path = instructDir + os.sep + 'clench.mp4'
        rightStim_path = instructDir + os.sep + 'poke.mp4'
        actionText_path = instructDir + os.sep + 'action_text.png' #obeject image with text
        gain_path = instructDir + os.sep + 'gain.png'
        noGain_path = instructDir + os.sep + 'no_gain.png'

        return dict(resp_pic=resp_pic,
                object_img=object_img,
                leftStim_path=leftStim_path,
                midStim_path=midStim_path,
                rightStim_path=rightStim_path,
                actionText_path=actionText_path,
                gain_path=gain_path,
                noGain_path=noGain_path)
    imagePaths = dict2class(setImagePath(instructDir))

    def instructParam(expInfo):
        # Number of pages
        numPages = 18
        return dict(numPages=numPages)
    instructInfo = dict2class(instructParam(expInfo))

    return dict(instructInfo=instructInfo,
                imagePaths=imagePaths)


class InstructPages(object):
    def __init__(self, taskObj, dispInfo, instructInfo, imagePaths):
        self.screen = taskObj.screen
        self.posHigh = [0, 0.6]
        self.posMid = [0, 0]
        self.posLow = [0, -0.6]
        self.height = 0.08
        self.color = 'white'
        self.wrapWidth = 1.8
        self.imageSize = 1.0
        self.imagePosL = [-0.5, -0.2]
        self.imagePosM = [0, -0.2]
        self.imagePosR = [0.5, -0.2]
        self.videoSize = (240, 135)
        self.videoPosL = [-230, -50]
        self.videoPosM = [0, -50]
        self.videoPosR = [230, -50]
        # Set up navigation images
        self.sizeNav = dispInfo.imageSize
        self.heightNav = self.height*(3/4)
        # Display information
        self.monitorX = dispInfo.monitorX
        self.monitorY = dispInfo.monitorY
        self.screenScaling = dispInfo.screenScaling
        # Set up navigation objects
        self.imagePaths = imagePaths
        self.navBack = visual.TextStim(win=self.screen,
                                       text="Press 'L' to go back",
                                       pos=[-0.7, -0.9],
                                       height=self.heightNav,
                                       color=self.color,
                                       wrapWidth=self.wrapWidth)
        self.navForward = visual.TextStim(win=self.screen,
                                          text="Press 'R' to go forward",
                                          pos=[0.7, -0.9],
                                          height=self.heightNav,
                                          color=self.color,
                                          wrapWidth=self.wrapWidth)

    def instruct_10(self):
        self.textTop = visual.TextStim(self.screen,
                                       text='Thank you for participating in this experiment. \n\n You will use your right hand to make actions during the task.\n\n Please put your right hand on the apparatus as shown. \n\n Your wrist should be on the fabric.',
                                       pos=self.posHigh,
                                       alignHoriz='center',
                                       height=self.height,
                                       color=self.color,
                                       wrapWidth=self.wrapWidth)

        self.respPic = visual.ImageStim(win=self.screen,
                                        image=self.imagePaths.resp_pic, # the apparatus image
                                        size=self.imageSize,
                                        pos=self.imagePosM)
        # Rescale images

        self.respPic.rescaledSize = rescaleStim(self.respPic, self.imageSize, self)
        self.respPic.setSize(self.respPic.rescaledSize)

        # Draw objects
        self.textTop.draw()
        self.respPic.draw()
        # Draw instruction navigation
        # self.navBack.draw()
        self.navForward.draw()
        # Flip screen
        self.screen.flip()
        return

    def instruct_11(self):
        self.textTop = visual.TextStim(self.screen,
                                       text="Now let's practice the hand gestures. \n\n Make the palm position and be ready.",
                                       alignHoriz='center',
                                       height=self.height,
                                       color=self.color,
                                       wrapWidth=self.wrapWidth)
        # Draw response keys
        self.navBack.draw()
        self.navForward.draw()
        # Draw objects
        self.textTop.draw()
        # Flip screen
        self.screen.flip()
        return

    # interactive page
    def instruct_12(self):
        self.textTop = visual.TextStim(win=self.screen,
                                       text='Try to Pinch now. \n\n The start position must be PALM!',
                                       alignHoriz='center',
                                       anchorVert='top',
                                       height=self.height,
                                       color=self.color,
                                       wrapWidth=self.wrapWidth)
        # Draw objects
        self.textTop.draw()
        # Flip screen
        self.screen.flip()
        return

    # interactive page
    def instruct_13(self):
        self.textTop = visual.TextStim(win=self.screen,
                                       text='Try to Clench now. \n\n The start position must be PALM!',
                                       alignHoriz='center',
                                       anchorVert='top',
                                       height=self.height,
                                       color=self.color,
                                       wrapWidth=self.wrapWidth)
        # Draw objects
        self.textTop.draw()
        # Flip screen
        self.screen.flip()
        return

    # interactive page
    def instruct_14(self):
        self.textTop = visual.TextStim(win=self.screen,
                                       text='Try to Poke now. \n\n The start position must be PALM!',
                                       alignHoriz='center',
                                       anchorVert='top',
                                       height=self.height,
                                       color=self.color,
                                       wrapWidth=self.wrapWidth)
        # Draw objects
        self.textTop.draw()
        # Flip screen
        self.screen.flip()
        return

    # interactive page
    def instruct_15(self):
        self.textTop = visual.TextStim(win=self.screen,
                                       text='Try to Pinch now. \n\n The start position must be PALM!',
                                       alignHoriz='center',
                                       anchorVert='top',
                                       height=self.height,
                                       color=self.color,
                                       wrapWidth=self.wrapWidth)
        # Draw objects
        self.textTop.draw()
        # Flip screen
        self.screen.flip()
        return

    # interactive page
    def instruct_16(self):
        self.textTop = visual.TextStim(win=self.screen,
                                       text='Try to Clench now. \n\n The start position must be PALM!',
                                       alignHoriz='center',
                                       anchorVert='top',
                                       height=self.height,
                                       color=self.color,
                                       wrapWidth=self.wrapWidth)
        # Draw objects
        self.textTop.draw()
        # Flip screen
        self.screen.flip()
        return

    # interactive page
    def instruct_17(self):
        self.textTop = visual.TextStim(win=self.screen,
                                       text='Try to Poke now. \n\n The start position must be PALM!',
                                       alignHoriz='center',
                                       anchorVert='top',
                                       height=self.height,
                                       color=self.color,
                                       wrapWidth=self.wrapWidth)
        # Draw objects
        self.textTop.draw()
        # Flip screen
        self.screen.flip()
        return

    def instruct_18(self):
        self.textTop = visual.TextStim(win=self.screen,
                                       text="Ok, the instructions are over.\n\n Let's do some practice.\n\n Please wait a moment for the experimenter.",
                                       pos=self.posMid,
                                       alignHoriz='center',
                                       height=self.height,
                                       color=self.color,
                                       wrapWidth=self.wrapWidth)
        # Draw objects
        self.textTop.draw()
        # Draw instruction navigation
        self.navBack.draw()
        self.navForward.draw()
        # Flip screen
        self.screen.flip()
        return
