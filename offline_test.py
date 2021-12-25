import cv2
import argparse
import os
from scripts import label_image as li
import random

time={}

class ConfusionMatrix:
    def __init__(self):
        self.label_names=[]
        self.video_name=[]
        self.overall_acc =[]
        self.specificity=[]
        self.sensitivity=[]
        self.precision=[]
        self.recall=[]
        self.fscore=[]
        self.overall_truth=[]
        self.overall_predict=[]

    def perf_measure(self,y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        
        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==0:
                TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                FN += 1

        return(TP, FP, TN, FN)

    def calculate(self, truth, pred, acc, label_name, video_name):
        self.overall_truth+=truth
        self.overall_predict+=pred
        self.label_names.append(label_name)
        self.video_name.append(video_name)
        if len(acc)!=0:
            self.overall_acc.append(sum(acc)/len(acc))
        else:
            self.overall_acc.append(0)
        TP, FP, TN, FN = self.perf_measure(truth, pred)
        
        if TP+FP==0:
            precision=0
        else:
            precision = round(TP/(TP+FP), 4)
        if TP+FN==0:
            recall=0
            self.sensitivity.append(0)
        else:
            recall =  round(TP/(TP+FN), 4)
            self.sensitivity.append(round(TP/(TP+FN), 4))
        if FP+TN==0:
            self.specificity.append(0)   
        else: 
            self.specificity.append(round(TN/(FP+TN), 4))
        self.precision.append(precision)
        self.recall.append(recall)
        if precision == 0 and recall == 0:
            fscore=0
        else:
            fscore=round((2* precision*recall)/(precision+recall), 4)
        self.fscore.append(fscore)
    
    def display(self):
        print("\n%-12s|%12s|%10s|%12s|%12s|%10s|%10s|%10s"%("Label Name", "Video Name", "Accuracy","Sensitivity", "Specificity","Precision", "Recall", "F-Score"))
        print("%-12s|%12s|%10s|%12s|%12s|%10s|%10s|%10s"%("="*12, "="*12, '='*10, '='*10, '='*10, '='*10, '='*10, '='*10))
        for label, video, acc, sen, spec, prec, recall, fscore in zip( self.label_names, self.video_name, self.overall_acc, self.sensitivity, self.specificity, self.precision, self.recall, self.fscore):
            print("%-12s|%12s|%10f|%12f|%12f|%10f|%10f|%10f"%(label, video[:10]+"..", acc, sen, spec, prec, recall, fscore))

    def overall_display(self):
        TP, FP, TN, FN = self.perf_measure(self.overall_truth, self.overall_predict)
        
        if TP+FP==0:
            precision=0
        else:
            precision = round(TP/(TP+FP), 4)
        if TP+FN==0:
            recall=0   
            sen=0         
        else:
            recall =  round(TP/(TP+FN), 4)  
            sen= recall          
        if FP+TN==0:
            spec=0 
        else: 
            spec=round(TN/(FP+TN), 4)
        if precision == 0 and recall == 0:
            fscore=0
        else:
            fscore=round((2* precision*recall)/(precision+recall), 4)
    
        print('\nSpecificity of videos is',spec)
        print('Sensitivity of',"videos is",sen)
        print('Precision of',"videos is",precision)
        print('Recall of',"videos is",recall)
        print('FScore of',"videos is",fscore)
        print('\nOverall Accuracy of',"videos is",round(sum(self.overall_acc)/len(self.overall_acc), 4))
    

def TestingVideoFiles(label_name, video_location, matrix):
    files = [f for f in os.listdir(video_location)]

    numb_vid=0
    time_local=0
    for file in files:        
        vidcap = cv2.VideoCapture(video_location+"/"+file)
        success, image = vidcap.read()
        acc=[]
        truth=[]
        predict=[]
        count=0
        print()
        while success:
            cv2.imwrite("./test1.jpg", image)  
            image_path =  "./test1.jpg"
            label,score, time_data  = li.findLabels_new(image_path)            
            truth.append(1)
            time_local+=time_data
            if label == label_name:
                predict.append(1)
            else:
                predict.append(0)
            acc.append(score)
            success, image = vidcap.read()
            os.remove(image_path)
            count+=1
            
            print("Calculating",label_name,count,end="\r")
        matrix.calculate(truth, predict, acc, label_name, file)
        time[label_name]=time_data
        numb_vid+=1
    
    

if __name__ == '__main__':
    matrix= ConfusionMatrix()
    for file in os.listdir("./offline_testing"):
        TestingVideoFiles(file,"./offline_testing/"+file, matrix)
    
    matrix.display()
    matrix.overall_display()
    for x in time:
        print('\nEvaluation time for '+x+': {:.3f}s\n'.format(time[x]))

    print('\nEvaluation time : {:.3f}s\n'.format(sum(time.values())))
    input("Press any key to exit...")