import numpy as np
import cv2
import time
import socket
from threading import Thread

#this is a generator that goes inside of a 2d array
def nested_for_loop(array1):
    array1_obj = iter(array1)
    array1_item = next(array1_obj)
    try:
        while True:
            for i in array1_item:
                yield i
            array1_item = next(array1_obj)
    except StopIteration:
        pass
    finally:
        del array1_obj


#Calculation part of poeple's tracking
def calculate(toleft, toright,w,h):
    global bounding_boxes
    global confidences,diff_members

    cv2.line(frame, (int(w / 2), 0), (int(w / 2), int(h)), (0, 255, 0), 3)  # passing line

    print(results)

    if len(results) > 0:
        for i in results.flatten():  # To make result 1d

            x_min,y_min,box_width,box_height = bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][2], bounding_boxes[i][3] #the detected person's coordinate features

            center = [(box_width / 2) + x_min, (box_height / 2) + y_min]  # To get center coordinates
            members.append(center)  # added to members

            # calculation

            if len(results) == len(previous_members):  # if result and previous members are equal then we can calculate nearest points
                for j in previous_members:
                    diff_members.append((center[0] - j[0]) ** 2 + (
                                center[1] - j[1]) ** 2)  # to calculate distance of each point's against previous points

                partner = diff_members.index(min(diff_members))  # to find each point's previous state

                if partner is None:  # eğer boşa devam et.
                    continue

                try:
                    if previous_members[partner][0] < w / 2 and center[0] > w / 2:  # if point pass to right, toright is increasing 1
                        toright += 1
                    elif previous_members[partner][0] > w / 2 and center[0] < w / 2:  # if point pass to left, toleft is increasing 1
                        toleft += 1
                except IndexError:  #if there is another person went in to frame, program is not calculating it
                    pass

            diff_members = []  # to recalculate distance between members, we're cleaning its inde

            text_box_current = '{}: {:.4f}'.format("head",
                                                   confidences[i])  # confidences

            t1 = Thread(target=cv2.putText,args=[frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2]) #to print confidences each head

            t2 = Thread(target=cv2.rectangle, args=[frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),      #to draw rectangles
                          (0, 255, 0), 2])
            t1.start()
            t2.start()
            t1.join()
            t2.join()

    return toleft,toright   #returning calculating results


def get_results(output_from_network):
    global bounding_boxes
    global confidences
    probability_minimum = 0.4
    threshold = 0.3

    for detected_objects in nested_for_loop(output_from_network):
        if detected_objects[5:][0] > probability_minimum:  #if confidence is higher then probability minimum
            x_center, y_center, box_width, box_height = detected_objects[0:4] * np.array([w, h, w, h]) #x,y,w,h values.


            #here is we are appending bounding boxes coordinates and confidences
            t1 = Thread(target=bounding_boxes.append,
                        args=[[int(x_center - (box_width / 2)), int(y_center - (box_height / 2)),
                              int(box_width), int(box_height)]])
            t2 = Thread(target=confidences.append, args =[float(detected_objects[5:][0])])

            t1.start()
            t2.start()
            t1.join()
            t2.join()


    return cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold) #here is we are returning ids







#setting camera features
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 426)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

h, w = None, None


#setting network
network = cv2.dnn.readNetFromDarknet("yolov3_test.cfg",
                                         "yolov3_train.weights")
#to get layers
layers_names_all = network.getLayerNames()
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]


#calculation summary variables
toleft = 0
toright = 0

#to hold people's coordinates
previous_members = []
partner = None
members = []
diff_members = []
while True:
    start = time.time() #to calculate fps
    _, frame = camera.read()
    frame = cv2.flip(frame,90) #to flip camera

    if w is None or h is None:
        h, w = frame.shape[:2] #to get height and weight

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320),
                                 swapRB=True, crop=False) #this is for getting blobs from frame (if you make (320,320) to (426,426) you can take higher accuracy but it degrades performance)

    network.setInput(blob)  # to send blobs to network
    output_from_network = network.forward(layers_names_output) # here is we are taking network output


    bounding_boxes = [] # to hold x,y,w,h features of bb's
    confidences = [] # to hold confidences

    results = get_results(output_from_network) #getting results
    toleft, toright = calculate(toleft,toright,w,h)


    #to print texts to frame
    cv2.putText(frame, f'ToLeft: {toleft}', (0, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f'ToRight: {toright}', (200, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    previous_members = members #here is we're assigning current frame's members to previous frame's members, to use it again in the next frame
    members = [] # to use members again, we're cleaning its inside
    
    #to display frames
    cv2.namedWindow('HHD360', cv2.WINDOW_NORMAL)
    cv2.imshow('HHD360', frame)

    #to quit from window when user pressed 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    print("frame per second : ",1/(time.time()-start))

#to close camera and windows that opened
camera.release()
cv2.destroyAllWindows()












