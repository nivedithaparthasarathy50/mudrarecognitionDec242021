from utils import detector_utils as detector_utils
import cv2
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import datetime
import argparse
from scripts import label_image as li
import imutils
import numpy as np

detection_graph, sess = detector_utils.load_inference_graph()

class ConfusionMatrix:
    def __init__(self):
        self.overall_truth=[]
        self.overall_predict=[]
        self.overall_acc=[]

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

    def calculate(self, truth, pred, acc):
        self.overall_truth+=truth
        self.overall_predict+=pred
        self.overall_acc+=acc
        
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
    



def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np, matrix):
    image= image_np
    label_names=["alapadmam", "kapitham", "katakamukam", "shikaram", "thripathakam"]
    acc=[]
    truth=[]
    predict=[]
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
            box = np.squeeze(boxes)
            ymin = int((box[0][0]*im_height))
            xmin = int((box[0][1]*im_width))
            ymax = int((box[0][2]*im_height))
            xmax = int((box[0][3]*im_width))
            rect  = image_np[ymin:ymax, xmin:xmax]
            cv2.imwrite("./images/hand"+str(0)+".jpg", rect)
            label,score = li.findLabels("./tf_files/retrained_graph.pb", "./images/hand"+str(0)+".jpg")
            text= label+":"+str(round(score, 2))
            truth.append(1)
            if label in label_names:
                predict.append(1)
            else:
                predict.append(0)
            acc.append(score)
            image=cv2.putText(image_np, text,(int(left), int(top)-4),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255),1,cv2.LINE_AA )
    matrix.calculate(truth, predict, acc)
    return image



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2
    matrix = ConfusionMatrix()

    #cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # draw bounding boxes on frame
        image_np=draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np, matrix)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            #cv2.imshow('Single-Threaded Detection',                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            frame = image_np
            frame = imutils.resize(frame, width=720, height=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow("Live Web Hand Tracking", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(25) & 0xFF == ord('q'):
            	matrix.overall_display() 
            	cv2.destroyAllWindows()
                input("Press any key to exit...")
            	break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
