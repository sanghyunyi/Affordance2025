from psychopy import visual, gui, data, core, event, logging, info
from psychopy.constants import *
import numpy as np
import os
from scipy.io import savemat
from config import *
from utils import *
from model import *
import cv2
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

def runBandit(expInfo, dispInfo, taskInfo, taskObj, keyInfo):
    taskInfo.subID = expInfo.SubNo
    taskInfo.dayNo = expInfo.dayNo

    outDir = expInfo.outDir + os.sep + str(expInfo.SubNo)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    outDir = expInfo.outDir + os.sep + str(expInfo.SubNo) + os.sep + str(expInfo.dayNo)
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    model = Model()
    qA = 0.2
    beta_list = [3.5]
    optimizer = optim.SGD([model.weight], lr=qA)



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

        sessionClock = core.Clock()
        skipSess = False

        sessionInfo = taskInfo.sessionInfo[sI]

        # Record disdaq time
        sessionInfo.__dict__.update({'tDisDaq':sessionClock.getTime()})

        # Proceed to trials
        start_time = sessionClock.getTime()
        progress_time = start_time

        for tI in range(session_length):
            # Record onset for start fixation
            sessionInfo.sessionOnsets.tPreFix[tI] = progress_time
            progress_time += sessionInfo.itiDur[tI]

            stimOfTrial = stimSeq[tI]
            condOfTrial = stimOfTrial.condition


            initTrial(tI, dispInfo, taskInfo, taskObj, sessionInfo, stimOfTrial, condOfTrial)
            # Run the trial
            simulation_input = (stimOfTrial, model, optimizer, beta_list)
            progress_time = runTrial(progress_time, simulation_input, sI, tI, taskObj, taskInfo, dispInfo, keyInfo, sessionInfo, sessionClock, condOfTrial, stream, outDir)

        # Saving the data
        if (expInfo.Version != 'pract' and sI == taskInfo.numSessions - 1):
            saveData(sI, expInfo, dispInfo, taskInfo, taskObj, keyInfo)

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
         resp_movIdx = 0
         sessionInfo.selectedMov[tI] = resp_movIdx
         pWin = sessionInfo.pWinOfMov[tI, resp_movIdx]
         isWin =  np.random.binomial(1,pWin,1).astype(bool)[0]
     elif (respKey == keyInfo.resp2):
         # Turn off the isi images
         resp_movIdx = 1
         sessionInfo.selectedMov[tI] = resp_movIdx
         pWin = sessionInfo.pWinOfMov[tI, resp_movIdx]
         isWin =  np.random.binomial(1,pWin,1).astype(bool)[0]
     elif (respKey == keyInfo.resp3):
         # Turn off the isi images
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
    return


def trialObj2Onehot(trialObj):
    path = trialObj.path.split(os.sep)
    typ = path[-2]
    name = int(path[-1].split('.')[0])
    file_idx = name
    if typ == 'Pinch':
        file_idx += 0
    elif typ == 'Clench':
        file_idx += 8
    else:
        file_idx += 16

    return file_idx


def runTrial(progress_time, simulation_input, sI, tI, taskObj, taskInfo, dispInfo, keyInfo, sessionInfo, sessionClock, condOfTrial, stream, outDir):
    # Flip screen and wait for response
    sessionInfo.sessionOnsets.tStim[tI] = stimOnset = progress_time

    response = None
    start_flag = False
    count = 0
    prediction = None
    hand_start_time = None
    process_duration = None

    stimOfTrial, model, optimizer, beta_list = simulation_input

    aff = np.array(stimOfTrial.affordance[:3])/200.
    stimOfTrial = trialObj2Onehot(stimOfTrial)

    resp, _ = actor(model, stimOfTrial, aff, beta_list)
    if resp == 0:
        response = [keyInfo.resp1]
    elif resp == 1:
        response = [keyInfo.resp2]
    else:
        response = [keyInfo.resp3]

    #response = np.random.choice([keyInfo.resp1, keyInfo.resp2, keyInfo.resp3], 1)

    waitTime = np.random.choice(np.linspace(0.5, 1,8, 100), 1)
    process_duration = np.random.choice(np.linspace(0.7, 1.0, 100), 1)

    progress_time += waitTime + process_duration

    # Process hand gesture response

    # Get response time to calculate RT below
    sessionInfo.sessionOnsets.tResp[tI] = respOnset = progress_time
    sessionInfo.sessionResponses.rt[tI] = waitTime = respOnset - stimOnset - process_duration
    # Which response was made
    isWin = None
    if keyInfo.resp1 in response:
        # Pinch gesture was made
        sessionInfo.sessionResponses.respKey[tI] = respKey = keyInfo.resp1
        isWin = computeOutcome(tI, dispInfo, taskInfo, taskObj, keyInfo, sessionInfo, respKey, condOfTrial)
        if isWin:
            outPath = taskObj.outGain.path
        else:
            outPath = taskObj.outNoGain.path
    elif keyInfo.resp2 in response:
        # Clench gesture was made
        sessionInfo.sessionResponses.respKey[tI] = respKey = keyInfo.resp2
        isWin = computeOutcome(tI, dispInfo, taskInfo, taskObj, keyInfo, sessionInfo, respKey, condOfTrial)
        if isWin:
            outPath = taskObj.outGain.path
        else:
            outPath = taskObj.outNoGain.path
    elif keyInfo.resp3 in response:
        # Poke gesture was made
        sessionInfo.sessionResponses.respKey[tI] = respKey = keyInfo.resp3
        isWin = computeOutcome(tI, dispInfo, taskInfo, taskObj, keyInfo, sessionInfo, respKey, condOfTrial)
        if isWin:
            outPath = taskObj.outGain.path
        else:
            outPath = taskObj.outNoGain.path

    # Update RL agent
    reward = torch.tensor([float(isWin)], dtype=torch.float64)
    Q = model(stimOfTrial)[resp]# + aff[resp]
    loss = F.mse_loss(reward, Q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    progress_time += sessionInfo.isiDur[tI]
    sessionInfo.sessionOnsets.tOut[tI] = fbOnset = progress_time

    progress_time += sessionInfo.fbDur[tI]

    sessionInfo.sessionOnsets.tPostFix[tI] = progress_time
    progress_time += taskInfo.trialInfo.maxRT - waitTime - process_duration

    return progress_time
