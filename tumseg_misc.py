import numpy as np
import scipy


'''
Evaluation functions
'''
def computeRates(Y, P, thress=0.5, cuda=True):
    '''
    Computes the TP, TN, FP and FN
    input:
        Y:          tensor of true labels
        P:          prediction of labels
        thress:     thresshold to be used for prediction
    Output:
        FP, TP, TN and FN
    '''
    label = Y == 1
    pred = P>=thress

    if cuda:
        TP = (label & pred).sum().float()
        TN = (~label & ~pred).sum().float()
        FP = (~label & pred).sum().float()
        FN = (label & ~pred).sum().float()
    else:
        TP = (label & pred).sum()
        TN = (~label & ~pred).sum()
        FP = (~label & pred).sum()
        FN = (label & ~pred).sum()
    
    return TP, TN, FP, FN

def precisionRecallFscore(TP, TN, FP, FN, detach=True):
    '''
    This function computes precision, recall and f1-score
    '''
    if TP + FP == 0:
        precision = TP
    else:
        precision = TP/(TP + FP)
        
    if TP + FN == 0:
        recall = TP
    else:
        recall = TP/(TP + FN)
        
    if precision + recall == 0:
        f1_score = precision
    else:
        f1_score = 2*(precision*recall)/(precision + recall)
    
    if detach:
        return precision.detach().cpu(), recall.detach().cpu(), f1_score.detach().cpu()
    else:
        return precision, recall, f1_score

def getPrecisionRecallFscore(Y, P, thress=0.5, cuda=True, detach=True):
    TP, TN, FP, FN = computeRates(Y,P,thress=thress, cuda=cuda)
    precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN, detach=detach)
    return precision, recall, f1_score

def getF1Score(Y, P, thress=0.5, cuda=True, detach=True):
    precision, recall, f1_score = getPrecisionRecallFscore(Y, P, thress=0.5, cuda=cuda, detach=True)
    return f1_score
    


def findSmallROIs(labels, size_thres=0.3):
    '''
    Args:
        labels (array): labels from ndimage.label.
        size_thres (float, optional): size threshold in percent of biggest roi to keep. Defaults to 0.3.

    Returns:
        label_small (list): labels to remove.

    '''
    roi_size = []
    num_features = np.max(labels)
    
    for i in range(1, num_features+1):
        size = (labels == i).sum()
        roi_size.append(size)
        
    roi_size = np.array(roi_size)
    # roi_size = roi_size/roi_size.max()*100
    roi_size = roi_size/roi_size.max()
    
    label_small = np.where(roi_size <= size_thres)[0]+1 # plus 1 to account for offset above
    
    return label_small

def findHighIntensity(labels, CT_in, thres=0.9, max_perc=0.1):
    '''
    Args:
        labels (array): labels from ndimage.label
        CT_in (array): CT image
        thress (float, optional): threshold for what is counted as high intensity. Defaults to 0.9.
        max_perc (float, optional): percentage of high intensity threshold. Defaults to 0.1.

    Returns:
        label_high (list): labels exceeding threshold for being high intensity.
    '''
    label_high = []
    num_features = np.max(labels)
    
    for i in range(1, num_features+1):
        vals = np.array((CT_in[labels == i]))
        q = np.quantile(vals, thres)
        perc = (vals >= q).sum()/len(vals)
        
        if perc >= max_perc:
            label_high.append(i)

    return np.array(label_high)


def postProcessROIs(output, CT_in, classify_thres=0.5, size_thres=0.3, remove_intensity=True, 
                    intensity_thres=0.99, intensity_perc_thres=0.1, verbose=False):
    
    proc_roi = output >= classify_thres
    labels, num_features = scipy.ndimage.label(proc_roi)
    
    if num_features == 0:
        print('WARNING: ROI is empty, returning an empty ROI')
        return proc_roi
    
    if remove_intensity:
        rm1 = findHighIntensity(labels, CT_in, thres=intensity_thres, max_perc=intensity_perc_thres)
    else:
        rm1 = np.array([])
    rm2 = findSmallROIs(labels, size_thres=size_thres)
    
    # collect unique rois to remove 
    rm = list(set(np.hstack((rm1,rm2))))
    
    if verbose:
        print('Found {} ROIs'.format(num_features))
        print('Kept {} ROIs'.format(num_features-len(rm)))
        print('-'*10)
        print('Removed {} intesity ROIs'.format(len(rm1)))
        print('Removed {} size ROIs'.format(len(rm2)))
        print('')
    
    # remove the ROIs labeled for exclusion
    proc_roi[np.isin(labels, rm)] = False
    
    if proc_roi.sum() != 0:
        return proc_roi
    else:
        return output >= classify_thres
   



