import numpy as np
import os 
import cv2
from shutil import rmtree
import utils

class Database(object):
    
    def __init__(self,Vars):         
        self.orig_database_path = Vars["orig_database_path"]
        
        self.db_new_path  = Vars["new_database_path"]
        
        self.img_resize = Vars["img_resize"]
        self.input_width = Vars["img_in_width"]
        self.input_height= Vars["img_in_height"]
        
        
        self.ratio = Vars["train_ratio"]
        self.shuffle = Vars["shuffle"]
        self.face_detection = Vars["face_detection"]
    
        self.original_age_labels = ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(25, 32)", "(38, 43)", "(48, 53)", "(60, 100)"]
        self.original_to_new_index = [0,         0,      1,          2,          3,          4,          5,          5 ]
        
        self.extra_age_labels = ['(38, 48)', '(38, 42)', '(8, 23)', '(27, 32)']
        self.extra_to_new_index = [ 4,           4,          -1,          3]
        
        self.age_labels = ["(0, 6)","(7, 14)","(15, 24)","(25, 34)","(35, 47)","(48, 100)" ]
        self.age_class_count = len( self.age_labels )
        
        self.sex_labels = ["m","f"]
        self.sex_class_count = len( self.sex_labels )
        
        self.info_files = ["fold_0_data.txt","fold_1_data.txt","fold_2_data.txt","fold_3_data.txt","fold_4_data.txt"]
  
    
        self.input_shape = (self.input_width,self.input_height,3)
    
        age_folder_name = "age"
        sex_folder_name = "sex"
        
        train_folder_name = "train"
        test_folder_name  = "test"
        
        self.db_age_folder_path = os.path.join(self.db_new_path, age_folder_name)
        self.db_sex_folder_path = os.path.join(self.db_new_path, sex_folder_name)       
        
        self.db_age_train_folder_path = os.path.join(self.db_age_folder_path, train_folder_name)
        self.db_age_test_folder_path = os.path.join(self.db_age_folder_path, test_folder_name)
        
        self.db_sex_train_folder_path  = os.path.join(self.db_sex_folder_path, train_folder_name)
        self.db_sex_test_folder_path  = os.path.join(self.db_sex_folder_path, test_folder_name)
        
        self.age_mean_image_path = os.path.join( self.db_age_folder_path, "mean_image.jpg" )
        self.sex_mean_image_path = os.path.join( self.db_sex_folder_path, "mean_image.jpg" )
    
    
    def read_info_files(self):
        all_infos =[]
        for info_file in self.info_files:
            with open( os.path.join(self.orig_database_path,info_file) ,"r") as file:
                line = file.readline()
                line = file.readline()
                while (line != ""):
                    line = line[:-1]
                    labels = line.split("\t")
        
                    img_dir = labels[0]
                    img_name = labels[1]
                    face_id = labels[2]
                    age = labels[3]
                    sex = labels[4]
                    
                    image_dir = os.path.join(self.orig_database_path,"aligned",img_dir)
                    image_name = "landmark_aligned_face."+face_id+"."+img_name
                    image_full_path = os.path.join(image_dir,image_name)
                    
                    all_infos.append( (face_id, image_full_path, age, sex)  )            
                    
                    line = file.readline()
        return all_infos
    
    def edit_all_infos(self, all_infos):
        infos_age = []
        [infos_age.append([]) for i in range(self.age_class_count)]
        
        infos_sex = []
        [infos_sex.append([]) for i in range(self.sex_class_count)]
        
        for i in range( len(all_infos) ):
            (face_id, path, age, sex) = all_infos[i]
            if age != 'None' and age != "" and age !=None:
                if age in self.original_age_labels:
                    new_age_index = self.original_to_new_index[self.original_age_labels.index(age)]
                elif age in self.extra_age_labels:
                    new_age_index = self.extra_to_new_index[self.extra_age_labels.index(age)]
                else:
                    if int(age) <= 6:
                        new_age_index = 0
                    elif int(age) <= 14:
                        new_age_index = 1
                    elif int(age) <= 24:
                        new_age_index = 2
                    elif int(age) <= 34:
                        new_age_index = 3
                    elif int(age) <= 47:
                        new_age_index = 4
                    else:
                        new_age_index = 5
                if new_age_index != -1:
                    age = self.age_labels[new_age_index]
                    infos_age[new_age_index].append( (path, age) )
             
            if sex == "m" :
                infos_sex[0].append( (path, sex) )
            elif sex == "f":
                infos_sex[1].append( (path, sex) )
                
        return infos_age, infos_sex
      
    def make_folders(self):
#       Creates Database Folders
#        - "database_name"
#               - "age"
#                   - "train"
#                       -(0, 6) ....
#                   - "test"    
#               - "sex"
#                   - "train"
#                   - "test"
        
        
        os.mkdir(self.db_new_path)
        
#       Age Folders
        os.mkdir(self.db_age_folder_path)
        
        os.mkdir(self.db_age_train_folder_path)
        os.mkdir(self.db_age_test_folder_path)
        
        for age in self.age_labels:
            os.mkdir(os.path.join(self.db_age_train_folder_path,age))
            os.mkdir(os.path.join(self.db_age_test_folder_path,age))
            
#       Sex Folders
        os.mkdir(self.db_sex_folder_path)
        
        os.mkdir(self.db_sex_train_folder_path)
        os.mkdir(self.db_sex_test_folder_path)
        
        for sex in self.sex_labels:
            os.mkdir(os.path.join(self.db_sex_train_folder_path,sex))
            os.mkdir(os.path.join(self.db_sex_test_folder_path,sex))
        
    def split_infos(self,infos,age=True):
        train = []
        test = []
        if age:
            class_count = self.age_class_count
        else:
            class_count = self.sex_class_count
        for i in range( class_count ) :
            train_size = int(len( infos[i] ) * self.ratio )
            for j in range( train_size ):
                train.append( infos[i][j] )
            for j in range(train_size,len(infos[i])):
                test.append( infos[i][j] )  
            
        return train, test
    
    def copy_all2(self,file_infos, dst_dir):
        if self.face_detection:
            from face_detection import FaceDetection
            fd = FaceDetection()
        
        classses = []
        counter=0
        mean_image = np.zeros(self.input_shape,np.float32)
        for path, class_feature in file_infos:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if self.face_detection:
                img = cv2.resize(img, (self.img_resize,self.img_resize), cv2.INTER_AREA)
                img = fd.crop_face(img) 
                if img is not None:
                    img = cv2.resize(img, (self.input_width,self.input_height), cv2.INTER_AREA)
                    if dst_dir is self.db_age_train_folder_path or dst_dir is self.db_sex_train_folder_path:
                        mean_image += img
                        counter +=1
                    cv2.imwrite( os.path.join(os.path.join( dst_dir, class_feature ), os.path.basename(path)), img)
                    classses.append( class_feature )
            else:
                img = cv2.resize(img, (self.input_width,self.input_height), cv2.INTER_AREA)
                if dst_dir is self.db_age_train_folder_path or dst_dir is self.db_sex_train_folder_path:
                    mean_image += img
                    counter+=1
                cv2.imwrite( os.path.join(os.path.join( dst_dir, class_feature ), os.path.basename(path)), img)
                classses.append( class_feature )
        
        
        if dst_dir is self.db_age_train_folder_path:
            mean_image /= counter
            mean_image = np.asarray(mean_image,np.uint8)
            cv2.imwrite( self.age_mean_image_path ,mean_image)
        elif dst_dir is self.db_sex_train_folder_path:
            mean_image /= counter
            mean_image = np.asarray(mean_image,np.uint8)
            cv2.imwrite( self.sex_mean_image_path ,mean_image )
        
        return classses
        
    def create_database(self):
        all_infoss = self.read_info_files()
        
        if self.shuffle:
            from random import shuffle
            shuffle(all_infoss)
        else:
            all_infoss.sort(key=lambda tup: int(tup[0]))
                
        age_infos, sex_infos = self.edit_all_infos(all_infoss)
        
        self.make_folders()

#       Copy Age Class Images        
        train_age,  test_age = self.split_infos(age_infos,age=True)
        age_train_occured_labels = self.copy_all2(train_age,self.db_age_train_folder_path)
        self.copy_all2(test_age,self.db_age_test_folder_path)
        
        age_classs_weight = self.calculate_class_weights(self.age_labels, age_train_occured_labels)
        
        utils.save_obj(age_classs_weight,self.db_age_folder_path)
        print("AGE CLASS WEIGHT")
        print(age_classs_weight)

#       Copy Sex Class Images        
        train_sex, test_sex = self.split_infos(sex_infos,age=False)
        sex_train_occured_labels = self.copy_all2(train_sex,self.db_sex_train_folder_path)
        self.copy_all2(test_sex,self.db_sex_test_folder_path)
        
        sex_classs_weight = self.calculate_class_weights(self.sex_labels, sex_train_occured_labels)
        
        utils.save_obj(sex_classs_weight,self.db_sex_folder_path)
        print("SEX CLASS WEIGHT")
        print(sex_classs_weight)
        
    def calculate_class_weights(self, labels, occured_labels):                
        from sklearn.utils import compute_class_weight        
        
        label_indexes = [labels.index(i) for i in occured_labels]
        
        class_weight_list = compute_class_weight('balanced', np.unique(label_indexes) ,label_indexes )
        
        class_weight = dict(zip(np.unique(label_indexes), class_weight_list))
        
        return class_weight
    
   
def main():
    from variables import JsonVariables
    variables = JsonVariables("input_variables.json", "model_variables.json")
    Vars = variables.vars    
    
    DB = Database(Vars)
    
    DB.create_database() 


if __name__ == "__main__":
    main()
    

    
    
    