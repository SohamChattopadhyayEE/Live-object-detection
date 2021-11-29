from flask import Flask, render_template, url_for, Response
import argparse

from utils.object_detection import object_detection as live_object_detection


parser = argparse.ArgumentParser(description="Object detection with OpenCV")
# Paths 
parser.add_argument('-wt','--model_weights',type=str,default='model_yolov3/yolov3.weights',
                    help= "Path to the pre-trained weights of the network")
parser.add_argument('-cfg','--model_cfg',type=str,default='model_yolov3/yolov3.cfg',
                    help= "Path to the .config file of the network")
parser.add_argument('-cls','--dataset_classes',type=str,default='coco.txt',
                    help= "Path to the classes file of the dataset")
parser.add_argument('-is_realtime','--is_realtime',action='store_false',
                    help="Is real time object detection is being performed")
parser.add_argument('-vid','--vid_path',type=str,default='video.mp4',
                    help= "Path to the video file")

# Parameters
parser.add_argument('-fd','--frame_dimension',type=int,default=320,
                    help= "The dimensions of the frames")
parser.add_argument('-conf_threh','--threshold_confidence',type=int,default=0.3,
                    help= "Threshold confidence for selecting an object")
args = parser.parse_args()


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/object_detection')
def object_detection():
    model = live_object_detection(args.model_weights, args.model_cfg, args.dataset_classes, args.is_realtime, 
                                    args.vid_path, args.frame_dimension, args.threshold_confidence)
    return Response(model.detect_object(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)


