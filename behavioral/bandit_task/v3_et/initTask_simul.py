from psychopy import visual, gui, data, core, event, logging, info
from psychopy.constants import *
import numpy as np
import os, pickle
from config import *
import pickle

np.random.seed(0)

def generateStims(expInfo, stim0, stim1, stim2):
    subID = expInfo.SubNo
    dayNo = expInfo.dayNo

    key = subID * 937 + dayNo * 509

    familiarity_balanced_mix = pickle.load(open(expInfo.designDir + os.sep + "familiarity_balanced_mix.pkl", "rb"))
    familiarity_balanced_mix = familiarity_balanced_mix[key % len(familiarity_balanced_mix)]

    # 0~4 in stim0, stim1 and 0~1 in stim2 are unfamialiar stimuli
    stim0_unfam = stim0[:5]
    stim0_fam = stim0[5:]
    stim1_unfam = stim1[:5]
    stim1_fam = stim1[5:]
    stim2_unfam = stim2[:2]
    stim2_fam = stim2[2:]

    stim0 = np.empty(8, dtype=object) # Pinch
    stim1 = np.empty(8, dtype=object) # Clench
    stim2 = np.empty(8, dtype=object) # Poke

    unfam_mix0 = familiarity_balanced_mix[0][0]
    fam_mix0 = familiarity_balanced_mix[0][1]
    idx = 0
    for i in unfam_mix0:
        stim0[i] = stim0_unfam[idx]
        idx += 1

    idx = 0
    for i in fam_mix0:
        stim0[i] = stim0_fam[idx]
        idx += 1

    unfam_mix1 = familiarity_balanced_mix[1][0]
    fam_mix1 = familiarity_balanced_mix[1][1]
    idx = 0
    for i in unfam_mix1:
        stim1[i] = stim1_unfam[idx]
        idx += 1

    idx = 0
    for i in fam_mix1:
        stim1[i] = stim1_fam[idx]
        idx += 1

    unfam_mix2 = familiarity_balanced_mix[2][0]
    fam_mix2 = familiarity_balanced_mix[2][1]
    idx = 0
    for i in unfam_mix2:
        stim2[i] = stim2_unfam[idx]
        idx += 1

    idx = 0
    for i in fam_mix2:
        stim2[i] = stim2_fam[idx]
        idx += 1

    return stim0, stim1, stim2

def distributeStim2Sess(expInfo, taskInfo, stim0, stim1, stim2, stim3):
    subID = expInfo.SubNo
    dayNo = expInfo.dayNo

    numSessions = expInfo.numSess

    stim = np.empty(numSessions, dtype=object)

    key = subID * 937 + dayNo * 509
    if key%2 == 0:
        for sI in range(numSessions):
            stimS = np.empty(taskInfo.stimPerSess, dtype=object)
            if sI == 0:
                stimS[0] = stim0[0]
                stimS[1] = stim0[1]
                stimS[2] = stim1[0]
                stimS[3] = stim2[0]
                stimS[0].pHighIdx = 0 # cong
                stimS[1].pHighIdx = 2 # incong
                stimS[2].pHighIdx = 1 # cong
                stimS[3].pHighIdx = 1 # incong
            elif sI == 1:
                stimS[0] = stim0[2]
                stimS[1] = stim1[1]
                stimS[2] = stim2[1]
                stimS[3] = stim0[3]
                stimS[0].pHighIdx = 0 # cong
                stimS[1].pHighIdx = 2 # incong
                stimS[2].pHighIdx = 2 # cong
                stimS[3].pHighIdx = 1 # incong
            elif sI == 2:
                stimS[0] = stim1[2]
                stimS[1] = stim1[3]
                stimS[2] = stim2[2]
                stimS[3] = stim0[4]
                stimS[0].pHighIdx = 1 # cong
                stimS[1].pHighIdx = 0 # incong
                stimS[2].pHighIdx = 2 # cong
                stimS[3].pHighIdx = 2 # incong
            elif sI == 3:
                stimS[0] = stim1[4]
                stimS[1] = stim2[3]
                stimS[2] = stim0[5]
                stimS[3] = stim1[5]
                stimS[0].pHighIdx = 1 # cong
                stimS[1].pHighIdx = 0 # incong
                stimS[2].pHighIdx = 0 # cong
                stimS[3].pHighIdx = 2 # incong
            elif sI == 4:
                stimS[0] = stim2[4]
                stimS[1] = stim0[6]
                stimS[2] = stim1[6]
                stimS[3] = stim2[5]
                stimS[0].pHighIdx = 2 # cong
                stimS[1].pHighIdx = 1 # incong
                stimS[2].pHighIdx = 1 # cong
                stimS[3].pHighIdx = 0 # incong
            elif sI == 5:
                stimS[0] = stim2[6]
                stimS[1] = stim2[7]
                stimS[2] = stim0[7]
                stimS[3] = stim1[7]
                stimS[0].pHighIdx = 2 # cong
                stimS[1].pHighIdx = 1 # incong
                stimS[2].pHighIdx = 0 # cong
                stimS[3].pHighIdx = 0 # incong

            stimS[2].isLowRew = True
            stimS[3].isLowRew = True
            stimS[0].condition = 0 # cong high
            stimS[1].condition = 1 # incong high
            stimS[2].condition = 2 # cong low
            stimS[3].condition = 3 # incong low
            stim[sI] = stimS
    else:
        for sI in range(numSessions):
            stimS = np.empty(taskInfo.stimPerSess, dtype=object)
            if sI == 0:
                stimS[0] = stim0[0]
                stimS[1] = stim0[1]
                stimS[2] = stim2[0]
                stimS[3] = stim1[0]
                stimS[0].pHighIdx = 0 # cong
                stimS[1].pHighIdx = 1 # incong
                stimS[2].pHighIdx = 2 # cong
                stimS[3].pHighIdx = 2 # incong
            elif sI == 1:
                stimS[0] = stim0[2]
                stimS[1] = stim2[1]
                stimS[2] = stim1[1]
                stimS[3] = stim0[3]
                stimS[0].pHighIdx = 0 # cong
                stimS[1].pHighIdx = 1 # incong
                stimS[2].pHighIdx = 1 # cong
                stimS[3].pHighIdx = 2 # incong
            elif sI == 2:
                stimS[0] = stim1[2]
                stimS[1] = stim0[4]
                stimS[2] = stim2[2]
                stimS[3] = stim1[3]
                stimS[0].pHighIdx = 1 # cong
                stimS[1].pHighIdx = 2 # incong
                stimS[2].pHighIdx = 2 # cong
                stimS[3].pHighIdx = 0 # incong
            elif sI == 3:
                stimS[0] = stim1[4]
                stimS[1] = stim1[5]
                stimS[2] = stim0[5]
                stimS[3] = stim2[3]
                stimS[0].pHighIdx = 1 # cong
                stimS[1].pHighIdx = 2 # incong
                stimS[2].pHighIdx = 0 # cong
                stimS[3].pHighIdx = 0 # incong
            elif sI == 4:
                stimS[0] = stim2[4]
                stimS[1] = stim1[6]
                stimS[2] = stim0[6]
                stimS[3] = stim2[5]
                stimS[0].pHighIdx = 2 # cong
                stimS[1].pHighIdx = 0 # incong
                stimS[2].pHighIdx = 0 # cong
                stimS[3].pHighIdx = 1 # incong
            elif sI == 5:
                stimS[0] = stim2[6]
                stimS[1] = stim2[7]
                stimS[2] = stim1[7]
                stimS[3] = stim0[7]
                stimS[0].pHighIdx = 2 # cong
                stimS[1].pHighIdx = 0 # incong
                stimS[2].pHighIdx = 1 # cong
                stimS[3].pHighIdx = 1 # incong

            stimS[2].isLowRew = True
            stimS[3].isLowRew = True
            stimS[0].condition = 0 # cong high
            stimS[1].condition = 1 # incong high
            stimS[2].condition = 2 # cong low
            stimS[3].condition = 3 # incong low
            stim[sI] = stimS

    # Alternating the even and odd indiced sessions
    session_mix_idx = pickle.load(open(expInfo.designDir + os.sep + "session_mix.pkl", "rb"))

    key = subID * 937 + dayNo * 509
    session_mix_idx = session_mix_idx[numSessions][key % len(session_mix_idx[numSessions])]

    sessions_stim = np.array([stim[i] for i in session_mix_idx], dtype=object)

    # For the practice
    stimS7 = np.empty(taskInfo.stimPerSess, dtype=object)
    stimS7[0] = stim3[0]
    stimS7[1] = stim3[1]
    stimS7[2] = stim3[2]
    stimS7[3] = stim3[3]
    stim_prac = np.array([stimS7], dtype=object)

    stim = np.append(sessions_stim, stim_prac, axis=0)

    return stim


def initTask(expInfo):
    ##### Task parameters properties ######
    # task properties
    numSessions = expInfo.numSess
    numMov = 3 # number of movements. (Pinch, Clench, Poke)
    numCond = 4 # number of conditions. (Congruent, incongruent)x (high,  low reward)
    numTrials = 20 * numCond * numSessions # 20 * numCond * numSessions
    numStim = expInfo.numSess * numCond # number of stimuli.
    # trial timing
    def trialParam(numSessions, numTrials):
        maxRT = 4
        isiMinTime = 1.#1.25
        isiMaxTime = 2.5#1.75
        fbMinTime = 1.
        fbMaxTime = 2.5
        if (expInfo.Modality == 'fMRI'):
            TR = 1.12 # Delete first 4 volumes
            disDaqTime = 4 * TR
        elif (expInfo.Modality == 'behaviour'):
            disDaqTime = 0
        minJitter = 1.
        maxJitter = 2.5
        trialsPerSess = numTrials // numSessions

        return dict(maxRT=maxRT,
                    isiMinTime=isiMinTime,
                    isiMaxTime=isiMaxTime,
                    fbMinTime=fbMinTime,
                    fbMaxTime=fbMaxTime,
                    disDaqTime=disDaqTime,
                    minJitter=minJitter,
                    maxJitter=maxJitter,
                    trialsPerSess=trialsPerSess,
                    numMov=numMov)
    trialInfo = dict2class(trialParam(numSessions, numTrials))
    # Win probabilities

    def taskParam():
        subID = expInfo.SubNo
        dayNo = expInfo.dayNo
        instructCond = str()
        pWinHigh = 0.8
        pWinLow = 0.2
        pWinHighLowRew = 0.4
        pWinLowLowRew = 0.1
        stimPerAff = numStim // numMov
        stimPerSess = numStim // numSessions
        outMag = 1
        trialDesigns = pickle.load(open(expInfo.designDir + os.sep + "trial_design.pkl", "rb"))
        sessionLengthsOffset = pickle.load(open(expInfo.designDir + os.sep + "session_lengths_offset.pkl", "rb"))
        return dict(subID=subID,
                    dayNo=dayNo,
                    instructCond=instructCond,
                    pWinHigh=pWinHigh,
                    pWinLow=pWinLow,
                    pWinHighLowRew=pWinHighLowRew,
                    pWinLowLowRew=pWinLowLowRew,
                    stimPerAff=stimPerAff,
                    stimPerSess=stimPerSess,
                    outMag=outMag,
                    trialDesigns=trialDesigns,
                    sessionLengthsOffset=sessionLengthsOffset)
    taskInfo = dict2class(taskParam())
    taskInfo.__dict__.update({'trialInfo':trialInfo,
                              'numSessions': numSessions,
                              'numTrials': numTrials,
                              'numCond': numCond})
    ###### Setting up the display structure #######

    # Keyboard info
    keyInfo = dict2class(keyConfig())

    # Stimuli set up

    subID = expInfo.SubNo
    dayNo = expInfo.dayNo

    stim0 = np.empty(8, dtype=object) # Pinch
    stim1 = np.empty(8, dtype=object) # Clench
    stim2 = np.empty(8, dtype=object) # Poke
    stim3 = np.empty(8, dtype=object) # This is for the practice

    stim0_dir = "StimuliSet" + os.sep + "Pinch"
    stim1_dir = "StimuliSet" + os.sep + "Clench"
    stim2_dir = "StimuliSet" + os.sep + "Poke"
    stim3_dir = "StimuliSet" + os.sep + "Neutral"

    # choose first 5 of stim0 and stim1 be unfamiliar and first 2 be unfamiliar of stim2

    initial_selection_mix = pickle.load(open(expInfo.designDir + os.sep + "initial_selection_mix.pkl", "rb"))
    initial_selection_mix = initial_selection_mix[subID % len(initial_selection_mix)]
    initial_selection_mix = initial_selection_mix[dayNo % len(initial_selection_mix)]

    for i, idx in enumerate(initial_selection_mix[0]):
        # Pinch image set
        stim0[i] = TrialObj(taskInfo, type='stim', pathToFile=expInfo.stimDir + os.sep + stim0_dir + os.sep + str(idx))
    for i, idx in enumerate(initial_selection_mix[1]):
        # Clench image set
        stim1[i] = TrialObj(taskInfo, type='stim', pathToFile=expInfo.stimDir + os.sep + stim1_dir + os.sep + str(idx))
    for i, idx in enumerate(initial_selection_mix[2]):
        # Poke image set
        stim2[i] = TrialObj(taskInfo, type='stim', pathToFile=expInfo.stimDir + os.sep + stim2_dir + os.sep + str(idx))

    for i, idx in enumerate(initial_selection_mix[3]):
        # Neutral image set for practice
        stim3[i] = TrialObj(taskInfo, type='stim', pathToFile=expInfo.stimDir + os.sep + stim3_dir + os.sep + str(idx))


    stim0, stim1, stim2 = generateStims(expInfo, stim0, stim1, stim2)

    stim = distributeStim2Sess(expInfo, taskInfo, stim0, stim1, stim2, stim3)

    # Outcome

    outGain = TrialObj(taskInfo, type='out',
            pathToFile=expInfo.stimDir + '/cb_' + str(expInfo.sub_cb) + '/gain' )
    outNoGain = TrialObj(taskInfo, type='noOut',
            pathToFile=expInfo.stimDir + '/cb_' + str(expInfo.sub_cb) + '/noGain')

    # Wrap objects into dictionary
    taskObj = dict2class(dict(stim=stim,
                           outGain=outGain,
                           outNoGain=outNoGain))

    # Initialize task variables
    taskInfo = initSessions(taskInfo, numSessions)
    screen = None
    dispInfo = None
    return screen, dispInfo, taskInfo, taskObj, keyInfo



class TrialObj(object):
    def __init__(self, taskInfo, type, pathToFile):
        # Static object parameters
        if (type == "stim"):
            self.path = pathToFile + '.png'
            self.path_magnified = pathToFile + '_magnified.png'
            self.path_fullshot = pathToFile + '_fullshot.png'
            # shown images after making a response
            self.respPath1 = pathToFile + '_resp1.png' #Pinch
            self.respPath2 = pathToFile + '_resp2.png' #Clench
            self.respPath3 = pathToFile + '_resp3.png' #Poke
            stimDir = os.sep + os.path.join(*pathToFile.split(os.sep)[:-2])
            affordance_score_dic = pickle.load(open(stimDir + os.sep + 'affordance_score_dic.pkl', 'rb'))
            pathToFile_list = pathToFile.split(os.sep)
            pathToFile_list[8] = 'v3'
            pathToFile = '/'.join(pathToFile_list)
            aff = affordance_score_dic[pathToFile] #[pinch score, clench score, poke score, palm score, familiarity]
            self.affordance = aff
            aff = aff[:3]
            self.highAffIdx = aff.index(max(aff))
            self.pHighIdx = None
            self.isLowRew = False
            self.condition = None
            # Initialize design containers
        elif (type == "out"):
            self.path = pathToFile + '.png'
        elif (type == "noOut"):
            self.path = pathToFile + '.png'

class Onsets(object):
    def __init__(self, numSessTrials):
        self.tPreFix = np.empty(numSessTrials)
        self.tStim = np.empty(numSessTrials)
        self.tResp = np.empty(numSessTrials)
        self.tOut = np.empty(numSessTrials)
        self.tPostFix = np.empty(numSessTrials)

class Responses(object):
    def __init__(self, numSessTrials):
        self.respKey = np.empty(numSessTrials)
        self.rt = np.empty(numSessTrials)


def initSessions(taskInfo, numSessions):
    # Set up the session-wise design
    sessionInfo = np.empty(taskInfo.numSessions, dtype=object)

    for sI in range(taskInfo.numSessions):
        numSessTrials = None

        average_session_lengths = taskInfo.trialInfo.trialsPerSess
        numCond = taskInfo.numCond
        if average_session_lengths == numCond:
            numSessTrials = numCond
        else:
            subID = taskInfo.subID
            dayNo = taskInfo.dayNo
            key = subID * 937 + dayNo * 509
            session_lengths_offset = taskInfo.sessionLengthsOffset
            session_lengths_offset = session_lengths_offset[key%len(session_lengths_offset)][sI]
            numSessTrials = average_session_lengths + numCond * session_lengths_offset

        # Trial design randomisations

        # ITI fixation
        itiDur = np.random.permutation(np.linspace(taskInfo.trialInfo.minJitter,
                                     taskInfo.trialInfo.maxJitter,
                                     numSessTrials))
        # action feedback duration
        isiDur = np.random.permutation(np.linspace(taskInfo.trialInfo.isiMinTime,
                                     taskInfo.trialInfo.isiMaxTime,
                                     numSessTrials))
        # reward feedback duration
        fbDur = np.random.permutation(np.linspace(taskInfo.trialInfo.fbMinTime,
                                     taskInfo.trialInfo.fbMaxTime,
                                     numSessTrials))

        # Store which stimuli was shown
        shownStim = np.empty(numSessTrials, dtype=object)
        # Store trial contition
        shownCond = np.empty(numSessTrials, dtype=int)
        # Store affordance score on each trial
        affordance = np.zeros((numSessTrials, taskInfo.trialInfo.numMov + 2),dtype=float) # + 2 is for palm and familiarity
        # Store pWin of each movement on each trial
        pWinOfMov = np.zeros((numSessTrials, taskInfo.trialInfo.numMov),dtype=float) # 3 is for each movement class
        # Store which movement was the selected movement
        selectedMov = np.zeros(numSessTrials,dtype=int)
        # Store whether the good (pWinHigh) option was chosen
        highChosen = np.zeros(numSessTrials,dtype=bool)
        # Initialize timing containers
        sessionOnsets = Onsets(numSessTrials)
        sessionResponses = Responses(numSessTrials)
        # Store whether reward was won in each trial
        isWin = np.empty(numSessTrials, dtype=bool)
        # Initialize payout container
        payOut = np.zeros(numSessTrials, dtype=float)
        # Store whether the trial is done or missed
        isDone = np.ones(numSessTrials,dtype=bool)
        # Flatten into class object
        sessionInfo[sI] = dict2class(dict(
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
            isDone=isDone))
    taskInfo.__dict__.update({'sessionInfo': sessionInfo})
    return(taskInfo)
