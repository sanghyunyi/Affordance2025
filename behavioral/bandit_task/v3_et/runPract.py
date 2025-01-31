from psychopy import visual, gui, data, core, event, logging, info
from psychopy.constants import *
import numpy as np
import os
from scipy.io import savemat
from config import *
from utils import *
import cv2
import itertools

np.random.seed(0)

class Onsets(object):
    def __init__(self,numPractTrials):
        self.tPreFix = np.empty(numPractTrials)
        self.tStim = np.empty(numPractTrials)
        self.tResp = np.empty(numPractTrials)
        self.tOut = np.empty(numPractTrials)
        self.tPostFix = np.empty(numPractTrials)

class Responses(object):
    def __init__(self,numPractTrials):
        self.respKey = np.empty(numPractTrials)
        self.rt = np.empty(numPractTrials)


def initPract(expInfo, taskInfo):
    # task properties
    numPractTrials = 20
    # Trial design randomisations
    itiDur = np.random.permutation(np.linspace(taskInfo.trialInfo.minJitter,
                                 taskInfo.trialInfo.maxJitter,
                                 numPractTrials))
    isiDur = np.random.permutation(np.linspace(taskInfo.trialInfo.isiMinTime,
                                 taskInfo.trialInfo.isiMaxTime,
                                 numPractTrials))
    fbDur = np.random.permutation(np.linspace(taskInfo.trialInfo.fbMinTime,
                                 taskInfo.trialInfo.fbMaxTime,
                                 numPractTrials))
    # Store which stimuli was shown
    shownStim = np.empty(numPractTrials, dtype=object)
    # Store trial contition
    shownCond = np.empty(numPractTrials, dtype=int)
    # Store affordance score on each trial
    affordance = np.zeros((numPractTrials, taskInfo.trialInfo.numMov + 2),dtype=float) # + 2 is for palm and familiarity
    # Store pWin of each movement on each trial
    pWinOfMov = np.zeros((numPractTrials, taskInfo.trialInfo.numMov),dtype=float) # 3 is for each movement class
    # Store which movement was the selected movement
    selectedMov = np.zeros(numPractTrials,dtype=int)
    # Store whether the good (pWinHigh) option was chosen
    highChosen = np.zeros(numPractTrials,dtype=bool)
    # Initialize timing containers
    sessionOnsets = Onsets(numPractTrials)
    sessionResponses = Responses(numPractTrials)
    # Store whether reward was won in each trial
    isWin = np.empty(numPractTrials, dtype=bool)
    # Initialize payout container
    payOut = np.zeros(numPractTrials, dtype=float)
    # Flatten into class object
    return dict(
        numPractTrials=numPractTrials,
        itiDur=itiDur,
        isiDur=isiDur,
        fbDur=fbDur,
        shownStim=shownStim,
        shownCond=shownCond,
        affordance=affordance,
        pWinOfMov=pWinOfMov,
        selectedMov=selectedMov,
        highChosen=highChosen,
        sessionOnsets=sessionOnsets,
        sessionResponses=sessionResponses,
        isWin=isWin,
        payOut=payOut)


def runPract(dispInfo, taskInfo, taskObj, keyInfo, practInfo):
    # Make all possible stim2condition mapping
    permutations = np.array(list(itertools.permutations(range(taskInfo.trialInfo.numMov), taskInfo.trialInfo.numMov))) #make [1,2,3] to [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    np.random.shuffle(permutations)

    # Initiate Video
    stream = cv2.VideoCapture(CAM_CODE)
    stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    while True:
        ret, img = stream.read()
        if ret:
            break

    img = rescale_frame(img, percent=10)
    _ = sender.send_image('HandPose', img)

    # Select stimuli for the session.
    stim = taskObj.stim[-1]

    sessionInfo = practInfo
    stimSeq = np.tile(stim, sessionInfo.numPractTrials//taskInfo.numCond)
    np.random.shuffle(stimSeq)

    # Wait for start confirmation
    while True:
        # Draw experimenter wait screen
        taskObj.readyExp.draw()
        taskObj.screen.flip()
        # Wait for starting confirmation response
        response = event.waitKeys(keyList=[keyInfo.instructDone, 'escape'])
        if keyInfo.instructDone in response:
            sessionClock = core.Clock()
            break
        elif 'escape' in response:
            print("Aborting program...")
            core.wait(2)
            core.quit()
    # Intitialize session
    taskObj.ITI.start(taskInfo.trialInfo.disDaqTime)
    taskObj.screen.flip()
    taskObj.startFix.setAutoDraw(False)
    taskObj.ITI.complete()  # Close static period

    # Initialize the reverseStatus to False (need participant to get 4 continuous correct)

    # Proceed to trials
    for tI in range(sessionInfo.numPractTrials):
        # Print trial number
        print('Trial No: ' + str(tI))
        # show fixation while loading
        trialStart = sessionClock.getTime()
        # Set up trial structure
        taskObj.ITI.start(sessionInfo.itiDur[tI])
        # Record onset for start fixation
        sessionInfo.sessionOnsets.tPreFix[tI] = sessionClock.getTime()
        # Show start fixation
        taskObj.startFix.setAutoDraw(True)
        taskObj.screen.flip()
        # Assign the current reward contingency
        taskObj.ITI.complete()

        stimOfTrial = stimSeq[tI]
        condOfTrial = stimOfTrial.condition

        initTrial(tI, dispInfo, taskInfo, taskObj, sessionInfo, stimOfTrial, condOfTrial)
        # Run the trial
        runTrial(tI, taskObj, taskInfo, dispInfo, keyInfo, sessionInfo, sessionClock, condOfTrial, stream)
        taskObj.ITI.complete()
        # Print trial timestamp
        print('Trial time ' + str(tI) + ': ' + str(sessionClock.getTime() - trialStart))
    # Show end ITI and save data
    taskObj.ITI.start(5)# Close screen
    taskObj.expEndFix.setAutoDraw(True)
    taskObj.screen.flip()
    # Print session time
    print('Session Time: ' + str(sessionClock.getTime() / 60))

    taskObj.expEndFix.setAutoDraw(False)
    taskObj.ITI.complete()

    stream.release()
    return




def computeOutcome(tI, dispInfo, taskInfo, taskObj, keyInfo, sessionInfo, respKey, condOfTrial):
     # Draw win and loss magnitudes
     outMag = taskInfo.outMag
     # Determine which stim was chosen
     if (respKey == keyInfo.resp1):
         # Turn off the isi images
         taskObj.firstResp.setAutoDraw(False)
         resp_movIdx = 0
         sessionInfo.selectedMov[tI] = resp_movIdx
         pWin = sessionInfo.pWinOfMov[tI, resp_movIdx]
         isWin =  np.random.binomial(1,pWin,1).astype(bool)[0]
     elif (respKey == keyInfo.resp2):
         # Turn off the isi images
         taskObj.secondResp.setAutoDraw(False)
         resp_movIdx = 1
         sessionInfo.selectedMov[tI] = resp_movIdx
         pWin = sessionInfo.pWinOfMov[tI, resp_movIdx]
         isWin =  np.random.binomial(1,pWin,1).astype(bool)[0]
     elif (respKey == keyInfo.resp3):
         # Turn off the isi images
         taskObj.thirdResp.setAutoDraw(False)
         resp_movIdx = 2
         sessionInfo.selectedMov[tI] = resp_movIdx
         pWin = sessionInfo.pWinOfMov[tI, resp_movIdx]
         isWin =  np.random.binomial(1,pWin,1).astype(bool)[0]

     sessionInfo.isWin[tI] = isWin
     # Record whether they chose the high value option
     if condOfTrial != 2: # congruent and incongruent condition
         sessionInfo.highChosen[tI] = True if (pWin == taskInfo.pWinHigh) else False
     else: # low reward condition
         sessionInfo.highChosen[tI] = None
         #sessionInfo.highChosen[tI] = True if (pWin == taskInfo.pWinHighLowRew) else False
     # Record the observed payOut
     sessionInfo.payOut[tI] = outMag/100 * 1 if isWin else outMag/100 * -1
     return isWin


def initTrial(tI, dispInfo, taskInfo, taskObj, sessionInfo, stimOfTrial, condOfTrial):
    # Initialize trial display drawings
    # End start-trial fixation
    taskObj.startFix.setAutoDraw(False)
    # Define stimulus and responses images for this trial
    # TODO how to use path_magnified?
    taskObj.midStim.image = stimOfTrial.path_fullshot
    taskObj.firstResp.image = stimOfTrial.respPath1
    taskObj.secondResp.image = stimOfTrial.respPath2
    taskObj.thirdResp.image = stimOfTrial.respPath3
    sessionInfo.shownStim[tI] = stimOfTrial.path
    sessionInfo.affordance[tI, :] = stimOfTrial.affordance
    sessionInfo.shownCond[tI] = condOfTrial
    #condOfTrial 0:congruent, 1:incongruent, 2:low congruent, 3:low incongruent
    if condOfTrial <= 2: # In the conditions of congruent and incongruent
        sessionInfo.pWinOfMov[tI, :] = taskInfo.pWinLow
        sessionInfo.pWinOfMov[tI, stimOfTrial.pHighIdx] = taskInfo.pWinHigh
    else: # In the condition of low reward
        sessionInfo.pWinOfMov[tI, :] = taskInfo.pWinLowLowRew
        sessionInfo.pWinOfMov[tI, stimOfTrial.pHighIdx] = taskInfo.pWinHighLowRew

    # Rescale images
    taskObj.midStim.rescaledSize = rescaleStim(taskObj.midStim, dispInfo.imageSize, dispInfo)
    taskObj.midStim.setSize(taskObj.midStim.rescaledSize)

    taskObj.firstResp.rescaledSize = rescaleStim(taskObj.firstResp, dispInfo.imageSize, dispInfo)
    taskObj.firstResp.setSize(taskObj.firstResp.rescaledSize)
    taskObj.secondResp.rescaledSize = rescaleStim(taskObj.secondResp, dispInfo.imageSize, dispInfo)
    taskObj.secondResp.setSize(taskObj.secondResp.rescaledSize)
    taskObj.thirdResp.rescaledSize = rescaleStim(taskObj.thirdResp, dispInfo.imageSize, dispInfo)
    taskObj.thirdResp.setSize(taskObj.thirdResp.rescaledSize)

    # Draw the stims
    taskObj.midStim.setAutoDraw(True)

    return


def runTrial(tI, taskObj, taskInfo, dispInfo, keyInfo, sessionInfo, sessionClock, condOfTrial, stream):
    # Flip screen and wait for response
    taskObj.screen.flip()
    sessionInfo.sessionOnsets.tStim[tI] = stimOnset = sessionClock.getTime()
    keyResponse = event.clearEvents()

    response = None
    start_flag = False
    count = 0
    prediction = None
    hand_start_time = None

    while (sessionClock.getTime() - stimOnset) <= taskInfo.trialInfo.maxRT:
        # process key response
        keyResponse = event.getKeys(keyList=[keyInfo.escapeKey])
        if keyInfo.escapeKey in keyResponse:
            core.wait(1)
            core.quit()

        # process hand gesture response
        # map 1: pinch, 2: clench, 3: poke
        ret, img = stream.read()
        if ret:
            img = rescale_frame(img, percent=10)
            t1 = sessionClock.getTime()
            prob = sender.send_image('HandPose', img)
            print(sessionClock.getTime()-t1)
            prob = prob.decode().split(",")
            prob = [float(p) for p in prob]
            print(prob)
            if start_flag == False:
                if max(prob) < 1.-sum(prob):
                    start_flag = True
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

    # Process hand gesture response
    if response:
        # Get response time to calculate RT below
        taskObj.ITI.start(sessionInfo.isiDur[tI])
        sessionInfo.sessionOnsets.tResp[tI] = respOnset = sessionClock.getTime()
        sessionInfo.sessionResponses.rt[tI] = waitTime = respOnset - stimOnset - process_duration
        # Which response was made
        if keyInfo.resp1 in response:
            # Pinch gesture was made
            sessionInfo.sessionResponses.respKey[tI] = respKey = keyInfo.resp1
            # Show response-specific ISI screen
            taskObj.midStim.setAutoDraw(False)
            taskObj.firstResp.setAutoDraw(True)
            taskObj.screen.flip()
            taskObj.firstResp.setAutoDraw(False)
            taskObj.ITI.complete()
            # Show outcome feedback
            taskObj.ITI.start(sessionInfo.fbDur[tI])
            isWin = computeOutcome(tI, dispInfo, taskInfo, taskObj, keyInfo, sessionInfo, respKey, condOfTrial)
            if isWin:
                outPath = taskObj.outGain.path
            else:
                outPath = taskObj.outNoGain.path
        elif keyInfo.resp2 in response:
            # Clench gesture was made
            sessionInfo.sessionResponses.respKey[tI] = respKey = keyInfo.resp2
            # Show response-specific ISI screen
            taskObj.midStim.setAutoDraw(False)
            taskObj.secondResp.setAutoDraw(True)
            taskObj.screen.flip()
            taskObj.secondResp.setAutoDraw(False)
            taskObj.ITI.complete()
            # Show outcome feedback
            taskObj.ITI.start(sessionInfo.fbDur[tI])
            isWin = computeOutcome(tI, dispInfo, taskInfo, taskObj, keyInfo, sessionInfo, respKey, condOfTrial)
            if isWin:
                outPath = taskObj.outGain.path
            else:
                outPath = taskObj.outNoGain.path
        elif keyInfo.resp3 in response:
            # Poke gesture was made
            sessionInfo.sessionResponses.respKey[tI] = respKey = keyInfo.resp3
            # Show response-specific ISI screen
            taskObj.midStim.setAutoDraw(False)
            taskObj.thirdResp.setAutoDraw(True)
            taskObj.screen.flip()
            taskObj.thirdResp.setAutoDraw(False)
            taskObj.ITI.complete()
            # Show outcome feedback
            taskObj.ITI.start(sessionInfo.fbDur[tI])
            isWin = computeOutcome(tI, dispInfo, taskInfo, taskObj, keyInfo, sessionInfo, respKey, condOfTrial)
            if isWin:
                outPath = taskObj.outGain.path
            else:
                outPath = taskObj.outNoGain.path

        # Resize outcome image
        taskObj.screen.flip()
        taskObj.midOut.image = outPath
        taskObj.midOut.rescaledSize = rescaleStim(taskObj.midOut, dispInfo.imageSize, dispInfo)
        taskObj.midOut.setSize(taskObj.midOut.rescaledSize)
        taskObj.midOut.setAutoDraw(True)
        taskObj.screen.flip()
        sessionInfo.sessionOnsets.tOut[tI] = fbOnset = sessionClock.getTime()
        taskObj.midOut.setAutoDraw(False)
        taskObj.ITI.complete()

        # Clear objects after presenting
        taskObj.firstResp.setAutoDraw(False)
        taskObj.secondResp.setAutoDraw(False)
        taskObj.thirdResp.setAutoDraw(False)
        taskObj.midStim.setAutoDraw(False)
        taskObj.ITI.complete()
        #  Present trial-end fixation
        taskObj.ITI.start(taskInfo.trialInfo.maxRT - waitTime)
        taskObj.endFix.setAutoDraw(True)
        taskObj.screen.flip()
        # Close out trial
        taskObj.endFix.setAutoDraw(False)
        sessionInfo.sessionOnsets.tPostFix[tI] = sessionClock.getTime()

    if not response:
        taskObj.ITI.start(sessionInfo.isiDur[tI] + sessionInfo.fbDur[tI])
        taskObj.midStim.setAutoDraw(False)
        taskObj.noRespErr.setAutoDraw(True)
        taskObj.screen.flip()
        taskObj.noRespErr.setAutoDraw(False)
        # Set onsets to nan
        sessionInfo.sessionOnsets.tResp[tI] = np.nan
        sessionInfo.sessionResponses.respKey[tI] = np.nan
        sessionInfo.sessionResponses.rt[tI] = np.nan
        waitTime = taskInfo.trialInfo.maxRT
        sessionInfo.sessionOnsets.tOut[tI] = np.nan
        # Set stim attributes to nan
        sessionInfo.isWin[tI] = np.nan
        sessionInfo.payOut[tI] = np.nan
        taskObj.ITI.complete()
        #  Present trial-end fixation
        taskObj.ITI.start(taskInfo.trialInfo.maxRT - waitTime)
        taskObj.endFix.setAutoDraw(True)
        taskObj.screen.flip()
        taskObj.endFix.setAutoDraw(False)
        sessionInfo.sessionOnsets.tPostFix[tI] = sessionClock.getTime()
    return
