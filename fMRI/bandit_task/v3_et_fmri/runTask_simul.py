import os
import fnmatch
from psychopy import visual, gui, data, core, event, logging, info
from psychopy.constants import *
import multiprocessing as mp
import time
homeDir = '/Users/YiSangHyun/Dropbox/Work/CurrentProjects/Affordance/banditTask/v3_et_fmri'
# Ensure that relative paths start from the same directory as this script
os.chdir(homeDir)
outDir = homeDir + os.sep + 'OutputSim'
stimDir = homeDir + os.sep + 'Images'
designDir = homeDir + os.sep + 'TaskDesigns'
# Create directories if they don't exist
if not os.path.exists(outDir):
    os.mkdir(outDir)

# Add dependencies
from config import *
from runInstruct import *
from runPract import *
from initTask_simul import *
from runBandit_simul import *

## Experiment start ##
# Store info about the experiment session
# Reference: allowable inputs
# SubNo. - has to be digits
# Version - test or debug or pract
# Modality - behaviour or fMRI
# Condition - money or juice
# numSess - number of sessions to run
# dayNo - day of testing

def simulate(SubNo):
    check = 0
    while check == 0:
        expName = 'Task'  # from the Builder filename that created this script
        expInfo = {'SubNo': SubNo,
                   'Version': 'debug', #test or debug
                   'Modality': 'behaviour', #behaviour or fmri
                   'numSess': 6,
                   'dayNo': 1,
                   'doInstruct': False,
                   'doPract': False,
                   'doTask': True}
        # Check to see there's no existing subject ID - if there is, repeat this dialogue box
        exist = False
        for file in os.listdir(outDir):
            if fnmatch.fnmatch(file, '*' + 'sub' + str(expInfo['SubNo']) + '*'):
                exist = True
        if (exist):
            check = 0
            print('That file already exists, try another subject number!')
        else:
            check = 1

    expInfo.update({'date': data.getDateStr(),  # add a simple timestamp
                    'expName': expName,
                    'homeDir': homeDir,
                    'outDir': outDir,
                    'stimDir': stimDir,
                    'designDir': designDir})
    print("expInfo: ", expInfo)
# Set up between-subject counterbalance
    expInfo = dict2class(counterbalance(expInfo))  # output is given by expInfo.sub_cb
#####  Task Section  #####
    taskClock = core.Clock()
# Initialize general parameters
    [screen, dispInfo, taskInfo, taskObj, keyInfo] = initTask(expInfo)
    instruct_initInfo = dict2class(initInstruct(expInfo, taskInfo, taskObj))
    practInfo = dict2class(initPract(expInfo, taskInfo))

    # Run instructions
    if (expInfo.doInstruct):
        runInstruct(expInfo, dispInfo, taskInfo, taskObj, keyInfo, instruct_initInfo)
    if (expInfo.doPract):
        runPract(dispInfo, taskInfo, taskObj, keyInfo, practInfo)
    # Run the taswk
    if (expInfo.doTask):
        runBandit(expInfo, dispInfo, taskInfo, taskObj, keyInfo)
    # Print out the final payment amount across the session(s)
    sessionPay = 0
    for sI in np.arange(taskInfo.numSessions):
        sessionPay += np.nansum(taskInfo.sessionInfo[sI].payOut)
    print('Payment: ' + str(sessionPay))

simNum = 100

if __name__ == "__main__":
    arg = list(range(1, 1+simNum))
    start = time.time()
    #for i in arg:
    #    simulate(i)
    with mp.Pool(8) as pool:
        pool.map(simulate, arg)
    print("==== Done ====")
    print("elapsed: ", (time.time() - start)/60)
