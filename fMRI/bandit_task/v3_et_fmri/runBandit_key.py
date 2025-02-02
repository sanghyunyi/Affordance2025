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
import pickle

np.random.seed(0)

def getIdxOfTwoAndOne(stim):
    # In each block, there are two stims with the same affordance.
    # So the Idx of Two is the indices of those two
    # Idx of one is the other 2 unique stims
    count = np.zeros(3)
    for i, s in enumerate(stim):
        count[s.highAffIdx] += 1
    stim_of_two = np.where(count == 2)
    indices_of_two_list = []
    indices_of_one_list = []
    for i, s in enumerate(stim):
        if s.highAffIdx == stim_of_two[0]:
            indices_of_two_list.append(i)
        else:
            indices_of_one_list.append(i)
    # return a A b c
    return indices_of_two_list + indices_of_one_list

def alphabet2idx(x):
    if x == "a":
        return 0
    elif x == "A":
        return 1
    elif x == "b":
        return 2
    else:
        return 3

def design2stimIdx(design, stimIdx):
    out = [stimIdx[alphabet2idx(x)] for x in design]
    return out

def getSessionLength(expInfo, taskInfo, sI):
    # counter balanced offsets random seq of [-1, 0, 1]
    session_lengths_offset = taskInfo.sessionLengthsOffset
    average_session_lengths = taskInfo.trialInfo.trialsPerSess
    numCond = taskInfo.numCond
    if average_session_lengths == numCond:
        return average_session_lengths
    else:
        subID = expInfo.SubNo
        session_lengths_offset = session_lengths_offset[subID%len(session_lengths_offset)][sI]
        average_session_lengths += numCond * session_lengths_offset
        return average_session_lengths

def getTrialDesign(expInfo, taskInfo, sI):
    trial_designs =taskInfo.trialDesigns
    subID = expInfo.SubNo
    trial_designs = trial_designs[(subID+sI)%len(trial_designs)]
    key = subID * 937 + sI * 509
    trial_design = trial_designs[key%len(trial_designs)]
    return trial_design

def randomizeBy4(seq):
    out = []
    for i in range(int(len(seq)/4)):
        sub_seq = seq[4*i:4*(i+1)]
        np.random.shuffle(sub_seq)
        out += sub_seq
    return out

def runBandit(expInfo, dispInfo, taskInfo, taskObj, keyInfo):
    outDir = expInfo.outDir + os.sep + str(expInfo.SubNo)
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    # Loop through sessions
    for sI in np.arange(taskInfo.numSessions):
        session_length = getSessionLength(expInfo, taskInfo, sI)

        stream = None

        # Select stimuli for the session.

        stim = taskObj.stim[sI]

        idxOf2stimsAnd1stim = getIdxOfTwoAndOne(stim)
        trial_design = getTrialDesign(expInfo, taskInfo, sI)
        stimSeqIdx = design2stimIdx(trial_design, idxOf2stimsAnd1stim)
        stimSeqIdx = stimSeqIdx[:session_length]
        stimSeq = [stim[i] for i in stimSeqIdx]
        stimSeq = np.array(randomizeBy4(stimSeq))


        skipSess = False
        # Wait for start confirmation
        if (expInfo.Modality == "behaviour"):
            while True:
                # Draw experimenter wait screen
                taskObj.readyExp.draw()
                taskObj.screen.flip()
                # Wait for starting confirmation response
                response = event.waitKeys(keyList=[keyInfo.instructDone, 'escape', keyInfo.escapeKey])
                if keyInfo.instructDone in response:
                    sessionClock = core.Clock()
                    break
                elif 'escape' in response:
                    print("Aborting program...")
                    core.wait(2)
                    core.quit()
                else:
                    print("Skip the session number "+str(sI+1))
                    skipSess = True
                    break

        elif (expInfo.Modality == 'fMRI'):
            while True:
                # Draw experimenter wait screen
                taskObj.waitExp.draw()
                taskObj.screen.flip()
                # Wait for starting confirmation response
                response = event.waitKeys(keyList=[keyInfo.instructDone, 'escape', keyInfo.escapeKey])
                if keyInfo.instructDone in response:
                    break
                elif 'escape' in response:
                    print("Aborting program...")
                    core.wait(2)
                    core.quit()
                else:
                    print("Skip the session number "+str(sI+1))
                    skipSess = True
                    break
            # Wait for scanner pulse if fMRI
            if not skipSess:
                while True:
                    # Draw pulse-wait screen
                    taskObj.scanPulse.draw()
                    taskObj.screen.flip()
                    # Wait for starting confirmation response
                    response = event.waitKeys(keyList=[keyInfo.pulseCode, 'escape'])
                    if keyInfo.pulseCode in response:
                        print('Received scanner trigger..')
                        # Initialize session clock
                        sessionClock = core.Clock()
                        taskObj.startFix.setAutoDraw(True)
                        break
                    elif 'escape' in response:
                        print("Aborting program...")
                        core.wait(2)
                        core.quit()
        if skipSess:
            continue

        # Intitialize session
        taskObj.ITI.start(taskInfo.trialInfo.disDaqTime)
        taskObj.screen.flip()
        taskObj.startFix.setAutoDraw(False)
        sessionInfo = taskInfo.sessionInfo[sI]
        taskObj.ITI.complete()  # Close static period
        # Record disdaq time
        sessionInfo.__dict__.update({'tDisDaq':sessionClock.getTime()})
        print('Disdaq time: ' + str(sessionClock.getTime()))

        # Proceed to trials
        for tI in range(session_length):
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
            runTrial(sI, tI, taskObj, taskInfo, dispInfo, keyInfo, sessionInfo, sessionClock, condOfTrial, stream, outDir)
            taskObj.ITI.complete()
            # Print trial timestamp
            print('Trial time ' + str(tI) + ': ' + str(sessionClock.getTime() - trialStart))
        # Show end ITI and save data
        taskObj.ITI.start(5)# Close screen
        taskObj.expEndFix.setAutoDraw(True)
        taskObj.screen.flip()
        # Print session time
        print('Session Time: ' + str(sessionClock.getTime() / 60))
        # Saving the data
        if (expInfo.Version != 'pract'):
            saveData(sI, expInfo, dispInfo, taskInfo, taskObj, keyInfo)
        taskObj.expEndFix.setAutoDraw(False)
        taskObj.ITI.complete()


    return


def saveData(sI, expInfo, dispInfo, taskInfo, taskObj, keyInfo):
    # Save ancilliary data
    outDir = expInfo.outDir + os.sep + str(expInfo.SubNo)
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    ancOutDir = outDir + os.sep + 'ancillary'
    if not os.path.exists(ancOutDir):
        os.mkdir(ancOutDir)

    save_obj(expInfo, ancOutDir + os.sep + 'sess' + str(sI+1) + '_expInfo')
    save_obj(dispInfo, ancOutDir + os.sep + 'sess' + str(sI+1) + '_dispInfo')
    save_obj(keyInfo, ancOutDir + os.sep + 'sess' + str(sI+1) + '_keyInfo')
    # Save data
    save_obj(taskInfo, outDir + os.sep + 'sess' + str(sI+1) + '_data')
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
     if condOfTrial < 2: # congruent and incongruent condition
         sessionInfo.highChosen[tI] = True if (pWin == taskInfo.pWinHigh) else False
     else: # low reward congruent and incongruent condition
         sessionInfo.highChosen[tI] = True if (pWin == taskInfo.pWinHighLowRew) else False
     # Record the observed payOut
     sessionInfo.payOut[tI] = outMag/100 * 1 if isWin else outMag/100 * 0
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
    #condOfTrial 0:high congruent, 1:high incongruent, 2:low congruent 3:low incongruent
    if condOfTrial < 2: # In the conditions of high reward
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


def runTrial(sI, tI, taskObj, taskInfo, dispInfo, keyInfo, sessionInfo, sessionClock, condOfTrial, stream, outDir):
    # Flip screen and wait for response
    taskObj.screen.flip()
    sessionInfo.sessionOnsets.tStim[tI] = stimOnset = sessionClock.getTime()
    keyResponse = event.clearEvents()

    response = None
    start_flag = False
    count = 0
    prediction = None
    hand_start_time = None
    process_duration = None


    while (sessionClock.getTime() - stimOnset) <= taskInfo.trialInfo.maxRT:
        # process key response
        keyResponse = event.getKeys(keyList=[keyInfo.resp1, keyInfo.resp2, keyInfo.resp3, 'escape'])
        if 'escape' in keyResponse:
            print("Aborting program...")
            core.wait(2)
            core.quit()
        elif keyInfo.resp1 in keyResponse or  keyInfo.resp2 in keyResponse or keyInfo.resp3 in keyResponse:
            response = keyResponse
            process_duration = 0.
            break
        else:
            continue

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
        taskObj.ITI.start(taskInfo.trialInfo.maxRT - waitTime - process_duration)
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
        sessionInfo.isDone[tI] = False
        taskObj.ITI.complete()
        #  Present trial-end fixation
        taskObj.ITI.start(taskInfo.trialInfo.maxRT - waitTime)
        taskObj.endFix.setAutoDraw(True)
        taskObj.screen.flip()
        taskObj.endFix.setAutoDraw(False)
        sessionInfo.sessionOnsets.tPostFix[tI] = sessionClock.getTime()
    return
