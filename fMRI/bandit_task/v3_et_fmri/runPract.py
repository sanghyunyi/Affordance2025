from psychopy import visual, gui, data, core, event, logging, info
from psychopy.constants import *
import numpy as np
import os
from scipy.io import savemat
from config import *
from utils import *
import cv2
import itertools
import threading

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
    # Store whether the trial is done or missed
    isDone = np.ones(numPractTrials,dtype=bool)
    # Store whether the trial is overrode
    isOverrode = np.zeros(numPractTrials,dtype=bool)
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
        payOut=payOut,
        isDone=isDone,
        isOverrode=isOverrode)


def runPract(dispInfo, taskInfo, taskObj, keyInfo, practInfo):
    # Make all possible stim2condition mapping
    permutations = np.array(list(itertools.permutations(range(taskInfo.trialInfo.numMov), taskInfo.trialInfo.numMov))) #make [1,2,3] to [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    np.random.shuffle(permutations)

    # Initiate Video
    stream = VideoCapture(CAM_CODE)
    while True:
        ret, img = stream.read('t')
        cv2.waitKey(int(1000/60))
        if ret:
            break

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
    del stream

    return




def computeOutcome(tI, dispInfo, taskInfo, taskObj, keyInfo, sessionInfo, respKey, condOfTrial):
     # Draw win and loss magnitudes
     outMag = taskInfo.outMag
     # Determine which stim was chosen
     if (respKey == keyInfo.resp1):
         # Turn off the isi images
         taskObj.respFix.setAutoDraw(False)
         resp_movIdx = 0
         sessionInfo.selectedMov[tI] = resp_movIdx
         pWin = sessionInfo.pWinOfMov[tI, resp_movIdx]
         isWin =  np.random.binomial(1,pWin,1).astype(bool)[0]
     elif (respKey == keyInfo.resp2):
         # Turn off the isi images
         taskObj.respFix.setAutoDraw(False)
         resp_movIdx = 1
         sessionInfo.selectedMov[tI] = resp_movIdx
         pWin = sessionInfo.pWinOfMov[tI, resp_movIdx]
         isWin =  np.random.binomial(1,pWin,1).astype(bool)[0]
     elif (respKey == keyInfo.resp3):
         # Turn off the isi images
         taskObj.respFix.setAutoDraw(False)
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
        keyResponse = event.getKeys(keyList=[keyInfo.resp1, keyInfo.resp2, keyInfo.resp3, 'escape'])
        if 'escape' in keyResponse:
            print("Aborting program...")
            core.wait(1)
            core.quit()
        elif keyInfo.resp1 in keyResponse or  keyInfo.resp2 in keyResponse or keyInfo.resp3 in keyResponse:
            response = keyResponse
            process_duration = 0.
            break
        else:
            continue

    # Process hand gesture response
    if response:
        if keyInfo.resp1 in response:
            print("===============")
            print("Response: Pinch")
            print("===============")
        elif keyInfo.resp2 in response:
            print("===============")
            print("Response: Clench")
            print("===============")
        elif keyInfo.resp3 in response:
            print("===============")
            print("Response: Poke")
            print("===============")
    if not response:
        print("===============")
        print("Response: None")
        print("===============")

    taskObj.ITI.start(sessionInfo.isiDur[tI])
    sessionInfo.sessionOnsets.tResp[tI] = respOnset = sessionClock.getTime()
    # Show post response fixation
    taskObj.midStim.setAutoDraw(False)
    taskObj.respFix.setAutoDraw(True)
    taskObj.screen.flip()

    new_respOnset = respOnset
    # Update the response when the classifier made error.
    while (sessionClock.getTime() - respOnset) < (sessionInfo.isiDur[tI] - 0.01):
        keyResponse = event.getKeys(keyList=[keyInfo.resp1, keyInfo.resp2, keyInfo.resp3, keyInfo.resp4, 'escape'])
        if keyInfo.resp1 in keyResponse or keyInfo.resp2 in keyResponse or keyInfo.resp3 in keyResponse or keyInfo.resp4 in keyResponse:
            response = keyResponse
            new_respOnset = sessionClock.getTime()
            if keyInfo.resp1 in response:
                print("Response Updated: Pinch")
                process_duration = 0
            elif keyInfo.resp2 in response:
                print("Response Updated: Clench")
                process_duration = 0
            elif keyInfo.resp3 in response:
                print("Response Updated: Poke")
                process_duration = 0
            elif keyInfo.resp4 in response:
                print("Response Updated: None")
                response = None
            sessionInfo.isOverrode[tI] = True


    taskObj.respFix.setAutoDraw(False)
    taskObj.ITI.complete()


    # Process hand gesture response
    if response:
        # Get response time to calculate RT below
        sessionInfo.sessionResponses.rt[tI] = waitTime = new_respOnset - stimOnset - process_duration
        # Which response was made
        if keyInfo.resp1 in response:
            # Pinch gesture was made
            sessionInfo.sessionResponses.respKey[tI] = respKey = keyInfo.resp1
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
        taskObj.respFix.setAutoDraw(False)
        taskObj.midStim.setAutoDraw(False)
        taskObj.ITI.complete()
        #  Present trial-end fixation
        taskObj.ITI.start(taskInfo.trialInfo.maxRT - (respOnset - stimOnset))
        taskObj.endFix.setAutoDraw(True)
        taskObj.screen.flip()
        # Close out trial
        taskObj.endFix.setAutoDraw(False)
        sessionInfo.sessionOnsets.tPostFix[tI] = sessionClock.getTime()

    if not response:
        taskObj.ITI.start(sessionInfo.fbDur[tI])
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
        sessionInfo.isDone[tI] = False
        taskObj.ITI.complete()
        #  Present trial-end fixation
        taskObj.ITI.start(taskInfo.trialInfo.maxRT - waitTime)
        taskObj.endFix.setAutoDraw(True)
        taskObj.screen.flip()
        taskObj.endFix.setAutoDraw(False)
        sessionInfo.sessionOnsets.tPostFix[tI] = sessionClock.getTime()
    return
