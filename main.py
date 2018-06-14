# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:58:47 2018

@author: FFM
"""
#def main():
#    print("Main Start")
#
#
#if __name__ == "__main__":
#    main()

from sklearn.metrics import confusion_matrix
import utils as myutils
from age_gender_model import AgeGenderModel
from variables import JsonVariables
from create_database import Database 

variables = JsonVariables("input_variables.json", "model_variables.json")
Vars = variables.vars
MV = variables.model_vars

DB = Database(Vars)

#DB.create_database()

agm = AgeGenderModel(MV, Vars, DB)
agm.create_model()

agm.init_image_data_generators()

#agm.train()


agm.save_model_weights()
agm.load_model_weights()

agm.test()
#
preds,true_oh = agm.predict_test()

acc_1off = myutils.accuracy_1off(preds,true_oh)

con_mat = confusion_matrix(
    true_oh.argmax(axis=1), preds.argmax(axis=1))

myutils.plot_confusion_matrix(
    con_mat, agm.class_labels,normalize=True, title="Confusion Matrix")

#print_preds(test_infos, preds, acc, my_acc, f1)
#
#
#j=0
#for i in range( len(data_test) ):
#    if np.argmax( preds[i] ) != np.argmax(data_test_y[i]):
#        j+=1
#        name= str(j)+" pred_"+self.class_labels[np.argmax(preds[i])] +" real_"+self.class_labels[np.argmax(data_test_y[i])]
#        name+=".jpg"
#        path=self.outputs_path+MV["name"]
#        imwrite(data_test[i],path, name )
#
#
#confusion_matrix_print_and_save(data_test_y,preds,self.class_labels,MV["name"])

#predict_image(img_path="test_img/6.jpg")
