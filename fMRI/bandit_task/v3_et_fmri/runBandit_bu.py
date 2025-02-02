from psychopy import visual, gui, data, core, event, logging, info
from psychopy.constants import *
import numpy as np
import os
from scipy.io import savemat
from config import *
from utils import *
import cv2, shutil
import itertools
import threading

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
        dayNo = expInfo.dayNo
        key = subID * 937 + dayNo * 509
        session_lengths_offset = session_lengths_offset[key%len(session_lengths_offset)][sI]
        average_session_lengths += numCond * session_lengths_offset
        return average_session_lengths

def getTrialDesign(expInfo, taskInfo, sI):
    trial_designs =taskInfo.trialDesigns
    subID = expInfo.SubNo
    dayNo = expInfo.dayNo
    trial_designs = trial_designs[(subID+dayNo+sI)%len(trial_designs)] # choosing between 0 and 1
    key = subID * 937 + sI * 509 + dayNo
    trial_design = trial_designs[key%len(trial_designs)]
    return trial_design

def randomizeBy4(seq):
    out = []
    for i in range(int(len(seq)/4)):
        sub_seq = seq[4*i:4*(i+1)]
        np.random.shuffle(sub_seq)
        out += sub_seq
    return out

def runBandit(expInfo, dispInfo, tracker, taskInfo, taskObj, keyInfo, io):
    # Assign the instruction condition (between subject)
    taskInfo.subID = expInfo.SubNo
    taskInfo.dayNo = expInfo.dayNo

    outDir = expInfo.outDir + os.sep + str(expInfo.SubNo)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    outDir = expInfo.outDir + os.sep + str(expInfo.SubNo) + os.sep + str(expInfo.dayNo)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        os.mkdir(outDir + os.sep + 'videos')


    stream = VideoCapture(CAM_CODE)
    # Loop through sessions
    for sI in np.arange(taskInfo.numSessions):
        session_length = getSessionLength(expInfo, taskInfo, sI)

        # Initiate Video
        while True:
            ret, img = stream.read('t')
            cv2.waitKey(int(1000/60))
            if ret:
                break

        # Select stimuli for the session.
        stim = taskObj.stim[sI]

        idxOf2stimsAnd1stim = getIdxOfTwoAndOne(stim)
        trial_design = getTrialDesign(expInfo, taskInfo, sI)
        stimSeqIdx = design2stimIdx(trial_design, idxOf2stimsAnd1stim)
        stimSeqIdx = stimSeqIdx[:session_length]
        stimSeq = [stim[i] for i in stimSeqIdx]
        stimSeq = np.array(randomizeBy4(stimSeq))

        #stimSeq = np.tile(stim, taskInfo.trialInfo.trialsPerSess // taskInfo.numCond)
        #np.random.shuffle(stimSeq)

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

                    tracker.sendMessage("EXPERIMENT ABORTED")
                    io.clearEvents()
                    tracker.setRecordingState(False)
                    #tracker.enableEventReporting(False) # End eye tracker data recording
                    tracker.setConnectionState(False)
                    io.quit() # Close iohub

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

                    tracker.sendMessage("EXPERIMENT ABORTED")
                    io.clearEvents()
                    tracker.setRecordingState(False)
                    #tracker.enableEventReporting(False) # End eye tracker data recording
                    tracker.setConnectionState(False)
                    io.quit() # Close iohub

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
                    response = event.waitKeys(keyList=[keyInfo.pulseCode, 'escape', keyInfo.escapeKey])
                    if keyInfo.pulseCode in response:
                        print('Received scanner trigger..')
                        # Initialize session clock
                        sessionClock = core.Clock()
                        taskObj.startFix.setAutoDraw(True)
                        break
                    elif 'escape' in response:
                        print("Aborting program...")
                        core.wait(2)

                        tracker.sendMessage("EXPERIMENT ABORTED")
                        io.clearEvents()
                        tracker.setRecordingState(False)
                        #tracker.enableEventReporting(False) # End eye tracker data recording
                        tracker.setConnectionState(False)
                        io.quit() # Close iohub

                        core.quit()
                    else:
                        print("Skip the session number "+str(sI+1))
                        skipSess = True
                        break
        if skipSess:
            continue

        # Video recording for a session start
        whole_sess_video_path = outDir + os.sep + 'videos' + os.sep + 'sess' + str(sI+1)

        stop_threads4whole_sess = False
        thread4whole_sess = threading.Thread(target=captureVideo, args=(lambda : stop_threads4whole_sess, whole_sess_video_path, sessionClock, stream, 's'))
        thread4whole_sess.start()

        # Intitialize session
        taskObj.ITI.start(taskInfo.trialInfo.disDaqTime)
        taskObj.screen.flip()
        taskObj.startFix.setAutoDraw(False)
        sessionInfo = taskInfo.sessionInfo[sI]
        taskObj.ITI.complete()  # Close static period
        # Record disdaq time
        sessionInfo.__dict__.update({'tDisDaq':sessionClock.getTime()})
        print('Disdaq time: ' + str(sessionClock.getTime()))
        print("Session " + str(sI+1) + " start")

        # Start getting data from the eye tracker
        #tracker.enableEventReporting(True)
        tracker.sendMessage("Session " + str(sI+1) + " start")
        tracker.setConnectionState(True)
        tracker.setRecordingState(True)

        # Proceed to trials
        #session_length = 3
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
            runTrial(sI, tI, taskObj, taskInfo, dispInfo, keyInfo, sessionInfo, sessionClock, condOfTrial, stream, outDir, tracker, io)
            taskObj.ITI.complete()
            # Print trial timestamp
            print('Trial time ' + str(tI) + ': ' + str(sessionClock.getTime() - trialStart))

        # Show end ITI and save data
        taskObj.ITI.start(5)# Close screen
        taskObj.expEndFix.setAutoDraw(True)
        taskObj.screen.flip()
        # Print session time
        print('Session Time: ' + str(sessionClock.getTime() / 60))

        # Save video of whole session
        stop_threads4whole_sess = True
        thread4whole_sess.join()

        # Saving the data
        if (expInfo.Version != 'pract'):
            saveData(sI, expInfo, dispInfo, taskInfo, taskObj, keyInfo)

            # Finish recording to eye tracker
            #tracker.enableEventReporting(False) # End eye tracker data recording
            tracker.sendMessage("Session " + str(sI+1) + " finished")
            io.clearEvents()

            tracker.setRecordingState(False)

        taskObj.expEndFix.setAutoDraw(False)
        taskObj.ITI.complete()

    stream.release()
    del stream

    tracker.setConnectionState(False) # Close and transfer eye-tracking data, then close down eye tracker connection
    io.quit()
    return

def moveETData(sI, expInfo):
    src_path = expInfo.homeDir + os.sep + 'et_data.EDF'
    target_path = expInfo.outDir + os.sep + str(expInfo.SubNo) + os.sep + str(expInfo.dayNo) + os.sep + 'sess' +str(sI+1) + '_et_data.EDF'
    shutil.move(src_path, target_path)

    return

def saveData(sI, expInfo, dispInfo, taskInfo, taskObj, keyInfo):
    outDir = expInfo.outDir + os.sep + str(expInfo.SubNo)
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    outDir = expInfo.outDir + os.sep + str(expInfo.SubNo) + os.sep + str(expInfo.dayNo)
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    # Save ancilliary data
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
    taskObj.midStim.image = stimOfTrial.path_fullshot
    sessionInfo.shownStim[tI] = stimOfTrial.path
    sessionInfo.affordance[tI, :] = stimOfTrial.affordance
    sessionInfo.shownCond[tI] = condOfTrial
    #condOfTrial 0:congruent, 1:incongruent, 2:low congruent 3:low incongruent
    if condOfTrial < 2: # In the conditions of congruent and incongruent
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


def runTrial(sI, tI, taskObj, taskInfo, dispInfo, keyInfo, sessionInfo, sessionClock, condOfTrial, stream, outDir, tracker, io):
    # Flip screen and wait for response
    taskObj.screen.flip()
    sessionInfo.sessionOnsets.tStim[tI] = stimOnset = sessionClock.getTime()
    keyResponse = event.clearEvents()

    tracker.sendMessage("sI: {}, tI: {}".format(sI+1, tI+1))

    response = None
    start_flag = False
    count = 0
    prediction = None
    hand_start_time = None
    process_duration = None

    # Video recording for a trial start
    single_trial_video_path = outDir + os.sep + 'videos' + os.sep + 'sess' + str(sI+1) +'_trial' + str(tI+1)

    stop_threads = False
    thread = threading.Thread(target=captureVideo, args=(lambda : stop_threads, single_trial_video_path, sessionClock, stream, 't'))
    thread.start()

    while (sessionClock.getTime() - stimOnset) <= taskInfo.trialInfo.maxRT:
        # process key response
        keyResponse = event.getKeys(keyList=[keyInfo.resp1, keyInfo.resp2, keyInfo.resp3, 'escape'])
        if 'escape' in keyResponse:
            print("Aborting program...")
            core.wait(2)

            tracker.sendMessage("EXPERIMENT ABORTED")
            io.clearEvents()
            tracker.setRecordingState(False)
            #tracker.enableEventReporting(False) # End eye tracker data recording
            tracker.setConnectionState(False)
            io.quit() # Close iohub

            core.quit()
        elif keyInfo.resp1 in keyResponse or  keyInfo.resp2 in keyResponse or keyInfo.resp3 in keyResponse:
            response = keyResponse
            process_duration = 0.
            break
        else:
            continue

    stop_threads = True
    thread.join()

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
