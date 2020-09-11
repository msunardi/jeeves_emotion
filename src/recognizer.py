#!/usr/bin/python

import rospy
from sensor_msgs.msg import Image as img_msg
from std_msgs.msg import String as string_msg
import numpy as np
from cv_bridge import CvBridge

import random as r
import time

import keras.backend as K
import sys
from deployment.tensorflow_detector import *
from deployment.utils import label_map_util
from deployment.utils import visualization_utils_color as vis_util
from deployment.video_threading_optimization import *
#from tensorflow_detector import *
#from utils import label_map_util
#from utils import visualization_utils_color as vis_util
#from video_threading_optimization import *

## OPTIONS ##

PATH_TO_CKPT = 'deployment/frozen_graphs/frozen_inference_graph_face.pb'
PATH_TO_CLASS = 'deployment/frozen_graphs/classificator_full_model.pb'
PATH_TO_REGRESS = 'deployment/frozen_graphs/regressor_full_model.pb'
label_map = label_map_util.load_labelmap('deployment/protos/face_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

window_not_set = True
bridge = CvBridge()
detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)

publisher = rospy.Publisher('/emotion', img_msg)
face_publisher = rospy.Publisher('/face', string_msg, queue_size=3)

emotion_buffer = []
emote = {'Neutral': 0, 'Happy': 0, 'Sad': 0, 'Afraid': 0, 'Surprised': 0, 'Disgusted': 0, 'Angry': 0, 'Contemptuous': 0}

def reset_emotion():
    global emotion_buffer, emote
    emotion_buffer = []
    emote = {'Neutral': 0, 'Happy': 0, 'Sad': 0, 'Afraid': 0, 'Surprised': 0, 'Disgusted': 0, 'Angry': 0, 'Contemptuous': 0}
    face_publisher.publish(string_msg('g'))

def jeeves_expression(emotions):
    print(emotions)
    if len(emotions) > 0 and emotions[0] != 'Neutral':
        print('SAD!')
        face_publisher.publish(string_msg('f'))
        face_publisher.publish(string_msg('w'))
        face_publisher.publish(string_msg('p'))
        face_publisher.publish(string_msg('3:6'))
    else:
        print("NEUTRAL")
        face_publisher.publish(string_msg('r'))


def compute_emote():
    global emotion_buffer, emote
    for e in emotion_buffer:
        if e not in emote.keys():
            emote[e] = 1
        else:
            emote[e] += 1

    return max(emote, key=emote.get)


def jeeves_expression2(emotion):
    print("FOO: {}".format(emotion))
    if emotion == 'Sad':
        print('SAD!')
        face_publisher.publish(string_msg('f'))
        face_publisher.publish(string_msg('w'))
        face_publisher.publish(string_msg('o'))
        face_publisher.publish(string_msg('3:6'))
        time.sleep(2)
        face_publisher.publish(string_msg('r'))
        face_publisher.publish(string_msg('f'))
        face_publisher.publish(string_msg('2:3'))
    elif emotion == 'Happy':
        print('HAPPY!')
        face_publisher.publish(string_msg('j'))
        if r.random() >= 0.5:
            face_publisher.publish(string_msg('q'))
        else:
            face_publisher.publish(string_msg('w'))
        face_publisher.publish(string_msg('3:2'))
    elif emotion == 'Surprised':
        print('SURPRISED!')
        if r.random() >= 0.5:
            face_publisher.publish(string_msg('l'))
        else:
            face_publisher.publish(string_msg('k'))
        face_publisher.publish(string_msg('q'))
        face_publisher.publish(string_msg('3:3'))
    else:
        print("NEUTRAL")
        face_publisher.publish(string_msg('r'))

def predict_camera(data):
    global emotion_buffer, emote
    # image = data.data
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    image = np.array(image, dtype=np.uint8)
    [h, w] = [data.height, data.width]

    # if window_not_set is True:
    #     cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
    #     window_not_set = False

    try:
        # image = cv2.flip(image, 1)
        print(image.shape)
        boxes, scores, classes, num_detections, emotions_print = detector.run(image)
        # print(scores)
        # print(classes)
        # print(emotions_print)
        text = "classes: {}".format(emotions_print)

        # jeeves_expression(emotions_print)

        if len(emotions_print) > 0 and len(emotion_buffer) < 10:
            emotion_buffer.append(emotions_print[0])
        elif len(emotion_buffer) >= 10:
            emotion = compute_emote()
            jeeves_expression2(emotion)

            # Reset emotion
            reset_emotion()

        cv2.putText(image, text, org=(25, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.35, color=(0, 255, 0))

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=1)

        # if window_not_set is True:
        #     cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
        #     window_not_set = False

        # cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)

        # Publish image
        pub_img = bridge.cv2_to_imgmsg(image, encoding='bgr8')
        publisher.publish(pub_img)

    except Exception as e:
        rospy.logerr("ERROR! {}".format(e))

def predict_from_camera(detector):
    print('Press q to exit')
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    window_not_set = True

    while True:
        # grab the frame from the threaded video stream
        image = vs.read()
        [h, w] = image.shape[:2]
        image = cv2.flip(image, 1)

        boxes, scores, classes, num_detections, emotions_print = detector.run(image)

        text = "classes: {}".format(emotions_print)
        cv2.putText(image, text, org=(25, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.35, color=(0, 255, 0))

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=1)

        if window_not_set is True:
            cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            window_not_set = False

        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

# subscriber = rospy.Subscriber('/usb_cam/image_raw', predict_camera, queue_size=10)

if __name__ == "__main__":
    rospy.init_node('emotion_recognizer', anonymous=True)
    rospy.Rate(10)
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    # detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)
    # predict_from_camera(detector)
    rospy.Subscriber('/usb_cam/image_raw', img_msg, predict_camera, queue_size=3)
    reset_emotion()
    rospy.spin()