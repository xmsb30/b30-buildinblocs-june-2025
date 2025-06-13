import tensorflow as tf, numpy as np, os, cv2, time, threading, pyautogui
from PIL import Image

start_samples = []
start_done = False
start_avg_acc = 0.0 
squint_threshold = 0.0 

squint_time = None 
nsquint_time = None

last_eye_detect = time.time()
last_face_detect = time.time() 

eyes_detected = False
zoom_size = [80, 90, 100, 110, 125, 150, 170, 200, 250, 300]
zoom_pos = 2 #100%
reading = False 

#If zoom in (check if can zoom in further else run the reading) If zoom out (check whether need to remove reading first)
def zoom(type="in"):
    global zoom_pos, reading
    
    if type == "in":
        if zoom_pos < (len(zoom_size) - 1):
            zoom_pos += 1
            pyautogui.hotkey('ctrl', '+')        
        if zoom_pos >= 4 and not reading:
            pyautogui.hotkey('alt', '0')
            pyautogui.hotkey('alt', 'P')
    elif type == "out":
        if zoom_pos > 0:
            zoom_pos -= 1
            pyautogui.hotkey('ctrl', '-')
        
        if zoom_pos < 4 and reading:
            pyautogui.hotkey('alt', 'O')
            pyautogui.hotkey('alt', '0')

#Initialize website
def reset_zoom():
    global zoom_pos
    pyautogui.hotkey('ctrl', '0')
    zoom_pos = 2
    if zoom_pos < 9:
        pyautogui.hotkey('alt', 'O')
        print("Website initialized")

script_dir = os.path.dirname(os.path.abspath('main.py'))
interpreter = tf.lite.Interpreter(os.path.join(script_dir, "eyes.tflite"))

#ML Model to detect faces
interpreter.allocate_tensors()
idetails = interpreter.get_input_details()
odetails = interpreter.get_output_details()

height = idetails[0]['shape'][1]
width = idetails[0]['shape'][2]
dtype = idetails[0]['dtype']

#ML Model to detect eyes
indices = {detail['name']: detail['index'] for detail in odetails}
boxes_idx = indices.get('detection_boxes', indices.get('TFLite_Detection_PostProcess', 0))
scores_idx = indices.get('detection_scores', indices.get('TFLite_Detection_PostProcess:2', 2))
num_detect_idx = indices.get('num_detections', indices.get('TFLite_Detection_PostProcess:3', 3))

face_cascade = cv2.CascadeClassifier(os.path.join(script_dir, "haarcascade_frontalface_default.xml"))
cam = cv2.VideoCapture(0)

while True:
    current_time = time.time()
    ret, frame = cam.read()   #Capture video from webcam
    
    if not ret:
        continue
    
    ###Adjusting webcam feed and running it through the face model
    new_frame = frame.copy()
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))   #(60, 60): Min resolution for face to be detected, minNeighbour (~ 2-6): Smaller less accurate

    has_faces = len(faces) > 0
    if has_faces:
        last_face_detect = current_time

    if not has_faces and (current_time - last_face_detect > 10.0):   ###If time since last face 10s, reset all variables
        start_avg_acc = 0.0
        squint_threshold = 0.0
        start_samples = []
        start_done = False
        squint_time = None
        nsquint_time = None
        reset_zoom()

        last_eye_detect = current_time
        eyes_detected = False

        continue

    eye_scores = []
    has_eyes = False

    if has_faces:
        for (face_x, face_y, face_w, face_h) in faces:
            cv2.rectangle(new_frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 0, 0), 2)   ###Display blue bounding box for face

            ###Break frame into grid of 6 and crop to center 4 columm with second and third zoom###
            crop_xmin = max(0, face_x + int(face_w / 6))
            crop_ymin = max(0, face_y + int(face_h / 6))
            crop_xmax = min(frame.shape[1], face_x + int(5 * face_w / 6))
            crop_ymax = min(frame.shape[0], face_y + int(3 * face_h / 6))


            region = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            cv2.rectangle(new_frame, (crop_xmin, crop_ymin), (crop_xmax, crop_ymax), (0, 0, 255), 2)   ###Display red box for general location range of eyes

            ###Adjusting webcam feed and running it through the eyes model with for cropped area
            pil_resized = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)).resize((width, height))
            input_data = np.expand_dims(np.array(pil_resized), axis=0)
            input_data = input_data.astype(np.uint8)

            interpreter.set_tensor(idetails[0]['index'], input_data)
            interpreter.invoke()

            scores = interpreter.get_tensor(scores_idx)
            num_detect = int(interpreter.get_tensor(num_detect_idx)[0])   #Too handle multiple eye detections

            for i in range(num_detect):
                if scores[0][i] >= 0.1:
                    box = interpreter.get_tensor(boxes_idx)[0][i]
                    ymin, xmin, ymax, xmax = box
                    
                    eye_left = crop_xmin + int(xmin * region.shape[1])
                    eye_top = crop_ymin + int(ymin * region.shape[0])
                    eye_right = crop_xmin + int(xmax * region.shape[1])
                    eye_bottom = crop_ymin + int(ymax * region.shape[0])

                    eye_area = (eye_right - eye_left) * (eye_bottom - eye_top)
                    crop_area = (crop_xmax - crop_xmin) * (crop_ymax - crop_ymin)

                    if eye_area > 0 and eye_area / crop_area <= 0.3:   ###Max area eyes can occur in red box
                        cv2.rectangle(new_frame, (eye_left, eye_top), (eye_right, eye_bottom), (0, 255, 0), 2)   ###Display green bounding box for eyes
                        cv2.putText(new_frame, f"eyes: {scores[0][i]:.2f}", (eye_left, eye_top - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        eye_scores.append(scores[0][i])

                        has_eyes = True

    if eye_scores:
        last_eye_detect = current_time

    if (has_eyes or (current_time - last_eye_detect <= 1.0)) and has_faces:   ###To account for blinking of eyes
        eyes_detected = True
    else:
        eyes_detected = False
        if current_time - last_eye_detect > 1.0 and has_faces:   ###To guarantee that user eyes not present
            squint_time = None
            nsquint_time = None

    if not start_done:
        if has_eyes:
            start_samples.extend(eye_scores)
            if len(start_samples) >= 30:    ###Initialize which is determine standard eyes accuracy
                print("start_samples", start_samples)
                start_avg_acc = np.mean(start_samples)
                squint_threshold = 0.85 * start_avg_acc   ###Determine if person squinting or not (threshold)
                start_done = True
                squint_time = None
                nsquint_time = current_time
    else:
        if eyes_detected and has_eyes:
            avg_confidence = np.mean(eye_scores)

            if avg_confidence < squint_threshold:
                if squint_time is None:
                    squint_time = current_time


                nsquint_time = None

                if current_time - squint_time >= 5.0:   ###If squinting for more than 5s then zoom in
                    squint_time = current_time

                    zoom("in")
                    print("Zoom in")
            elif avg_confidence >= squint_threshold:
                if nsquint_time is None:
                    nsquint_time = current_time

                squint_time = None

                if current_time - nsquint_time >= 30.0:   ###If not squinting for more than 30 then zoom out if possible
                    nsquint_time = current_time

                    zoom("out")
                    print("Zoom out")

        elif current_time - last_eye_detect > 10.0:   ###To check if user is gone for good, then restart
            start_avg_acc = 0.0
            start_samples = []
            start_done = False

            squint_threshold = 0.0
            
            squint_time = None
            nsquint_time = None

    cv2.imshow('SmartSquint', new_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()