import requests
import json
import os
from dotenv import load_dotenv
import base64 
import cv2
import psycopg2
from PIL import Image
from io import BytesIO
import face_recognition
import numpy as np
from threading import Thread, Event
import time 
import json
import progressbar
from numpyencoder import NumpyEncoder
from uuid import uuid4
load_dotenv() 


def preparedb():
    if os.getenv("PSQL_DSN") is None: 
        raise RuntimeError("ENV SQL_DSN not set")
    conn = psycopg2.connect(dsn=os.getenv("PSQL_DSN"))
    cur = conn.cursor()
    return cur, conn

def query(sql, vars = ()): 
    cur, conn = preparedb()
    cur.execute(sql, vars)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

def execute(sql, vars = ()): 
    cur, conn = preparedb()
    cur.execute(sql+" RETURNING id", vars)
    id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return id

def image_to_data_url(frame, image_format='jpeg'):

    # Encode image to memory buffer
    success, encoded_image = cv2.imencode(f'.{image_format}', frame)
    if not success:
        raise ValueError("Could not encode the image to the specified format")

    # Convert buffer to base64
    base64_image = base64.b64encode(encoded_image).decode('utf-8')

    # Create data URL
    data_url = f'data:image/{image_format};base64,{base64_image}'

    return data_url

def face_encoding(frame): 
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_location = face_recognition.face_locations(frame)
    if len(face_location) == 0 : 
        return list([]),face_location
    
    image_encoding = face_recognition.face_encodings(rgb_image)

    if len(image_encoding) > 0 : 
        return image_encoding,face_location
    else:
        return list([]),face_location

def matching_faces(knownencoding, checkencoding, frame, facelocation): 
    global label_list
    results = face_recognition.compare_faces(knownencoding, checkencoding, 0.5)
     # Check if any match is found
    
    finaldata = []
    
    for idx,item in enumerate(results): 
        if item == True: 
            checkifexist = list(filter(lambda x: x[0] == label_list[idx], finaldata))
            if len(checkifexist) > 0:
                finaldata = list(filter(lambda x: x[0] != label_list[idx], finaldata))
                finaldata.append((label_list[idx], checkifexist[0][1]+1))
            else:  
                finaldata.append((label_list[idx], 1))
    
    
    if len(finaldata) > 0: 
        max_data = max(finaldata, key=lambda x: x[1])
        print("Matching face is :: ",max_data)
        name = max_data[0]

        top, right, bottom, left = facelocation
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        dataurl = image_to_data_url(frame)
        print(dataurl)
        # url = "https://url to send telegram"

        # payload = json.dumps({
        #     "foo": "bar",
        # })
        # headers = {
        #     'Content-Type': 'application/json',
        #     'Authorization': 'Bearer token'
        # }

        # response = requests.request("POST", url, headers=headers, data=payload)

        # print(response.text)

def face_recognition_runtime(frame, knownencoding): 
    encodings,facelocation = face_encoding(frame)
    if encodings is None: 
        pass
    for idx,encoding in enumerate(encodings): 
        matching_faces(knownencoding, encoding, frame, facelocation[idx])
        


def trainImage(event): 
    registered_face = query("select id, label, reference_id from registered_faces")
    
    for item in registered_face:
        id,label,_ = item
        print("Train "+label)
        
        image_data = query("select id,data from registered_faces_data where trained_face_id is null and registered_face_id = %s", (id,))
        
        pbar = progressbar.ProgressBar().start()
        i = 0
        for data in image_data: 
            i = i+1
            pbar.update((i/len(image_data))*100)
            _, byteimage = data
            image = Image.open(BytesIO(byteimage))
            np_image = np.array(image)
            encoding_data = face_encoding(np_image)
            if encoding_data is None: 
                continue

            for encoding in encoding_data : 
                idtrainedface = execute("insert into trained_face(id,registered_face_id,data) values (%s, %s, %s)", (str(uuid4()),id,json.dumps(encoding, cls=NumpyEncoder)))
                execute("update registered_faces_data set trained_face_id = %s where registered_face_id = %s", (idtrainedface, id))
                
                if event.is_set():
                    print('The thread was stopped prematurely.')
                    break
        pbar.finish()


label_list = None
encoding_list = None
def trainImageFromDB(event): 
    print("get encoding from database")
    labeled_image = []
    registered_face = query("select id, label, reference_id from registered_faces")
    for item in registered_face:
        id,label,_ = item
        print("Train "+label)
        image_data = query("select data from trained_face where registered_face_id = %s", (id,))
        
        for data in image_data: 
            encoding, = data
            
            labeled_image.append((label, np.array(encoding)))
                
            if event.is_set():
                print('The thread was stopped prematurely.')
                break

    
    global encoding_list
    global label_list

    encoding_list = list(map(lambda x: x[1], labeled_image))
    
    label_list = list(map(lambda x: x[0], labeled_image))


def recognitionFn(event, uri): 
    while(True):
        if checkframe[uri] is not None: 
            facedetection = face_recognition.face_locations(checkframe[uri])
            if len(facedetection) > 0 and encoding_list is not None : 
                print("Face Detected")
                face_recognition_runtime(checkframe[uri], encoding_list)

        if event.is_set():
            print('The thread was stopped prematurely.')
            break
        
checkframe = {}
def captureRTSP(event, uri): 
    global checkframe
    print(checkframe)
    cap = cv2.VideoCapture(uri)
    while(True):
        ret, frame = cap.read()
        
        if os.getenv("PREVIEW") == "1" :
            cv2.imshow("RTSP "+uri, frame)

        
        checkframe[uri] = frame
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            event.set()
            break


    cap.release()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def main():
    
    # train_event = Event()
    # Thread(target=trainImage, args=(train_event,)).start()
    
    event = Event()
    trainImage(event)
    trainImageFromDB(event)

    

    rtsp_list = []

    if os.getenv("RTSP_URI") is not None: 
        rtsp_list.append(os.getenv("RTSP_URI"))
    else:
        print("ENV RTSP_URI not set")
        pass

    # setup frame
    for uri in rtsp_list: 
        checkframe[uri] = None

    for uri in rtsp_list: 
        Thread(target=recognitionFn, args=(event,uri,)).start()
        Thread(target=captureRTSP, args=(event,uri,)).start()

    


if __name__ == '__main__':

    main()