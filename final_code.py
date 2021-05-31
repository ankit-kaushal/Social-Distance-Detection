# importing all the required modules
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
from tkinter import *
from PIL import ImageTk
from playsound import playsound
import os


def viddetect():
    # base path to YOLO directory
    print("[INFO]  Dataloader")
    MODEL_PATH = "./tiny"

    MIN_CONF = 0.3  # We will use it as if it is above 30% the it is good detection
    NMS_THRESH = 0.5   # we will use it to reduce no of bounding boxes, just go on reducing its values

    # boolean indicating if NVIDIA CUDA GPU should be used
    USE_GPU = False

    # here we have defined the minimum safe distance (in pixels) that two people can be from each other
    MIN_DISTANCE = 100

    print("[INFO]  Detect people function")


    def detect_people(frame, net, ln, personIdx=0):
        # by this we find height, width of our image
        (H, W) = frame.shape[:2]
        results = []

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []  # this is the first list which will contain x,y,width and height
        centroids = []  # this list will contain all the class id's
        confidences = []    # this list will contain confidence values

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # first remove first 5 elements and to find heighest probabilities
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filtering detections by (1) ensuring that the object detected was a person and (2) that the minimum confidence is met
                if classID == personIdx and confidence > MIN_CONF:
                    # scaling the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # using the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # updating our list of bounding box coordinates, centroids, and confidences
                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extracting the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # updating our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)

        # return the list of results
        return results


    print("[INFO] Yolo configuration")

    # loading the COCO class labels our YOLO model was trained on

    labelsPath = './yolo-coco/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    print("Yolo detection category : ", LABELS[0])

    # deriving the paths to the YOLO weights and model configuration
    weightsPath = './tiny/yolov3-tiny.weights'
    configPath = './tiny/yolov3-tiny.cfg'

    # loading our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # determining only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initializing the video stream and pointer to output video file
    print("[INFO] accessing video stream...")
    cap = cv2.VideoCapture('PETS2009.avi')
    writer = None

    # looping over the frames from the video stream
    while True:
        (grabbed, frame) = cap.read()   # this will give the image

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # resizing the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
                                personIdx=LABELS.index("person"))
        # initializing the set of indexes that violate the minimum social distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:

            # extracting all centroids from the results and computing the
            # Euclidean distances between all pairs of the centroids

            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # looping over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # here we are checking if two persons are less that MIN_DISTANCE apart or not
                    if D[i, j] < MIN_DISTANCE:
                        # updating our violation set with the indexes of the centroid pairs
                        violate.add(i)
                        violate.add(j)
                        #playsound("tune.mp3")



        # looping over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extracting the bounding box and centroid coordinates, then
            # initializing the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)
            # if the index pair exists within the violation set, then update the color
            if i in violate:
                color = (0, 0, 255)
            # drawing (1) a bounding box around the person and (2) the centroid coordinates of the person,

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            #cv2.circle(frame, (cX, cY), 5, color, 1)

        # displaying the total number of person and total number of social distancing violations on the output frame

        text = "Total Person: {}".format(len(results))
        cv2.putText(frame, text, (10, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (23, 0, 74), 3)
        text = "Person violating Social Distancing: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (102, 0, 51), 3)

        # checking to see if the output frame should be displayed to our screen

        cv2.imshow("Frame", frame)  # to display the image

        key = cv2.waitKey(1) & 0xFF # delay it for 1 millisecond
        if key == ord("q"):
            break

        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('result.avi', fourcc, 25,
                                     (frame.shape[1], frame.shape[0]), True)
        # if the video writer is not None, write the frame to the output
        # video file
        if writer is not None:
            writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

def camdetect():
    # base path to YOLO directory
    print("[INFO]  Dataloader")
    MODEL_PATH = "./tiny"

    MIN_CONF = 0.3  # We will use it as if it is above 30% the it is good detection
    NMS_THRESH = 0.3   # we will use it to reduce no of bounding boxes, just go on reducing its values

    # boolean indicating if NVIDIA CUDA GPU should be used
    USE_GPU = False

    # here we have defined the minimum safe distance (in pixels) that two people can be from each other
    MIN_DISTANCE = 100

    print("[INFO]  Detect people function")


    def detect_people(frame, net, ln, personIdx=0):
        # by this we find height, width of our image
        (H, W) = frame.shape[:2]
        results = []

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []  # this is the first list which will contain x,y,width and height
        centroids = []  # this list will contain all the class id's
        confidences = []    # this list will contain confidence values

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # first remove first 5 elements and to find heighest probabilities
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filtering detections by (1) ensuring that the object detected was a person and (2) that the minimum confidence is met
                if classID == personIdx and confidence > MIN_CONF:
                    # scaling the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # using the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # updating our list of bounding box coordinates, centroids, and confidences
                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extracting the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # updating our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)

        # return the list of results
        return results


    print("[INFO] Yolo configuration")

    # loading the COCO class labels our YOLO model was trained on

    labelsPath = './yolo-coco/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    print("Yolo detection category : ", LABELS[0])

    # deriving the paths to the YOLO weights and model configuration
    weightsPath = './tiny/yolov3-tiny.weights'
    configPath = './tiny/yolov3-tiny.cfg'

    # loading our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # determining only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initializing the video stream and pointer to output video file
    print("[INFO] accessing camera...")
    cap = cv2.VideoCapture(0)
    writer = None

    # looping over the frames from the video stream
    while True:
        (grabbed, frame) = cap.read()   # this will give the image

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # resizing the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
                                personIdx=LABELS.index("person"))
        # initializing the set of indexes that violate the minimum social distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:

            # extracting all centroids from the results and computing the
            # Euclidean distances between all pairs of the centroids

            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # looping over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # here we are checking if two persons are less that MIN_DISTANCE apart or not
                    if D[i, j] < MIN_DISTANCE:
                        # updating our violation set with the indexes of the centroid pairs
                        violate.add(i)
                        violate.add(j)
                        #playsound("tune.mp3")



        # looping over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extracting the bounding box and centroid coordinates, then
            # initializing the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)
            # if the index pair exists within the violation set, then update the color
            if i in violate:
                color = (0, 0, 255)
            # drawing (1) a bounding box around the person and (2) the centroid coordinates of the person,

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            #cv2.circle(frame, (cX, cY), 5, color, 1)

        # displaying the total number of person and total number of social distancing violations on the output frame

        text = "Total Person: {}".format(len(results))
        cv2.putText(frame, text, (10, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (23, 0, 74), 3)
        text = "Person violating Social Distancing: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (102, 0, 51), 3)

        # checking to see if the output frame should be displayed to our screen

        cv2.imshow("Frame", frame)  # to display the image

        key = cv2.waitKey(1) & 0xFF # delay it for 1 millisecond
        if key == ord("q"):
            break

        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('result.avi', fourcc, 25,
                                     (frame.shape[1], frame.shape[0]), True)
        # if the video writer is not None, write the frame to the output
        # video file
        if writer is not None:
            writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

class Show:
    def __init__(self,root):
        self.root = root
        self.root.title("Social Distance Detection")
        self.root.geometry("1199x600+100+50")
        self.root.resizable(False, False)
        # ============================#
        self.bg = ImageTk.PhotoImage(file="front.jpg")
        self.bg_image = Label(self.root, image=self.bg).place(x=0, y=0, relwidth=1, relheight=1)
        # ==============================#
        Frame_show = Frame(self.root, bg="#e8f4f8")
        Frame_show.place(x=150, y=150, height=300, width=700)

        title = Label(Frame_show, text="Social Distance Detection", font=("Impact", 40), fg="#191970",
                      bg="#e8f4f8").place(x=40, y=10)
        subtitle = Label(Frame_show, text="Click any of the options below",
                         font=("times new roman", 20, "bold"), fg="#1c2841", bg="#e8f4f8").place(x=40, y=120)
        Show_btn = Button(Frame_show, text="Detect through Video", fg="white", bg="#191970", font=("times new roman", 20, "bold"),
                           command=viddetect).place(x=60, y=180)
        Show_btn2 = Button(Frame_show, text="Detect through Camera", fg="white", bg="#191970", font=("times new roman", 20, "bold"),
                           command=camdetect).place(x=360, y=180)

root=Tk()
obj=Show(root)
root.mainloop()