import os
import matplotlib.pylab as plt
import numpy as np
from  itertools import product
import cv2
import time
import pickle

def accuracy_1off(predictions, true_classes):    
    accuracy = 0
    for i in range( len( predictions ) ):
        true_index = true_classes[i].argmax()
        pred_index = predictions[i].argmax()
        if abs( true_index - pred_index ) <= 1:
            accuracy += 1
    accuracy /= len(predictions)
    return accuracy

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def save_obj(obj, path ):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path ):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

def log(log_txt):
    with open("outputs/LOG.txt","a") as log:
        log_time = time.ctime()
        log.write(str(log_time)+" | "+log_txt+"\n")

def log_preds(log_txt,folder_name):
    if not os.path.exists("outputs/"+folder_name):
        os.makedirs("outputs/"+folder_name,0o777)
    with open("outputs/"+folder_name+"/predictions.txt","a") as log:
        log.write(log_txt+"\n")

def accuracy_precision_recall_f1score(con_mat):
    TN = 0
    FN = np.sum(np.sum(con_mat,axis=0) - np.diagonal(con_mat))
    FP = np.sum(np.sum(con_mat,axis=1) - np.diagonal(con_mat))
    TP = np.sum(np.diagonal(con_mat))
    
    print(TN)
    print(FN)
    print(FP)
    print(TP)
    

def plots(ims, path, name, figsize=(40,20), rows=10, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = ims * 255
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=12)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    if not os.path.exists(path):
        os.makedirs(path)
    f.savefig( os.path.join(path, name + ".png" ) )
        
def confusion_matrix_print_and_save(con_mat,labels,path, name, normalize=True):
    fig = plt.figure()
    plot_confusion_matrix(con_mat,classes=labels,normalize=normalize)
    plt.show()    
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig( os.path.join(path, name + ".png" ) )

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#def imshow(img,title=""):
#    cv2.imshow(title,img)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
#    
#def imwrite(img_float,path,name):
#    img = img_float * 255
#    img = np.asarray(img,np.uint8)
#    cv2.imwrite(path+"/"+name,img)
#    
#def imread(path):
#    img = cv2.imread(path)
#    img = np.asarray(img,np.float32)
#    img /= 255
#    return img