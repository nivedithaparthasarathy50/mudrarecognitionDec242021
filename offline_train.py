import cv2
import argparse
import os

data={"label_name":[],"no_of_video":[], "overall_fps":[]}

def TrainingVideoFiles(video_location, video_name):
    files = [f for f in os.listdir(video_location)]
    data['label_name'].append(video_name)
    data['no_of_video'].append(len(files))
    fps=0
    
    try:
        os.mkdir("tf_files/mudras/"+video_name)
    except:
        print("Directory Already Exists")
    for index,file in enumerate(files):        
        vidcap = cv2.VideoCapture(video_location+"/"+file)
        success, image = vidcap.read()
        count = 1
        while success:
            cv2.imwrite("./tf_files/mudras/%s/image%d_%d.jpg" % (video_name,index, count), image)    
            success, image = vidcap.read()
            print('Saved image',video_name,index,"-", count, end="\r")
            count += 1
        fps+=count
    data['overall_fps'].append(fps)
            
if __name__ == '__main__':
    for file in os.listdir("./offline_training"):
        TrainingVideoFiles("./offline_training/"+file, file)
    
    print("\n\n%12s|%12s|%12s"%("Label Name", "No of Videos", "Overall Fps"))
    print("%12s|%12s|%12s"%("="*12,"="*12, "="*12))
    for label, vid, fps in zip(data['label_name'],data['no_of_video'], data['overall_fps']):
        print("%-12s|%12d|%12d"%(label, vid, fps))
    

    input("Press any key to exit...")


