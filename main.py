import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import paddleocr
from paddleocr import PaddleOCR
import easyocr
from datetime import datetime
import urllib
from urllib.request import urlopen

from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
    return {"Hello":"World"}

tkens={}

@app.get('/api/init') # img_path="./IMG_1.jpg"
def primaryFunc(img_path):
    url = img_path
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) # The image object
    # print(type(image))
    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    b1 = curr_datetime+'.jpg'
    # print(b1)
    cv2.imwrite(b1, image)
    image = 'C:/Users/sheri/bill-ocr-api/'+b1
    # read image
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # dictionary to store cropped coordinates
    itemList = {
        "image_path": img_path,
        "meter_reading": None,
        "meter_id": None
    }

    # dictionary to store extracted meter reading & id via OCR
    Tokens = {
        "image_path": img_path,
        "roi_coordinates": [], # saving roi coordinates
        "meter_reading": [],
        "meter_id": []
    }

    # select ROI function for two different objects in a loop

    for item in itemList.keys():
        if item != "image_path":
            print("Select the coordinates for", item)

            # select ROI function
            scale_percent = 30
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img_res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            roi = cv2.selectROI(img_res)

            # print rectangle points of selected roi
            Tokens['roi_coordinates'].append(roi)
            print(roi)

            # Crop selected roi from raw image
            roi_cropped = img_res[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            scale_percent = 200
            width = int(roi_cropped.shape[1] * scale_percent / 100)
            height = int(roi_cropped.shape[0] * scale_percent / 100)
            dim = (width, height)
            img_res = cv2.resize(roi_cropped, dim, interpolation = cv2.INTER_AREA)

            # show cropped image
            cv2.imshow(item, img_res)

            # save the cropped image
            itemList[item] = img_res

            # Get the current date and time from system
            # and use strftime function to format the date and time.
            curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

            # Split the picture path into root and extension
            newfileName = item + ".jpg"
            splitted_path = os.path.splitext(newfileName)

            # Add the current date time between root and extension
            modified_picture_path = splitted_path[0] + curr_datetime + splitted_path[1]

            cv2.imwrite(modified_picture_path, img_res)

            #hold window
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if item == 'meter_reading':
                # Initialize PaddleOCR
                ocr = PaddleOCR()

                # Perform OCR on the image
                result = ocr.ocr(img_res)

                print(result)
                output_text = ' '.join(item[1][0] for item in result[0])

                # Print the output text
                text = output_text
                Tokens['meter_reading'].append(text)
            else:
                #img = cv2.imread(img_res)
                gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.bitwise_not(thresh)
                reader = easyocr.Reader(['en'])
                result = reader.readtext(thresh)
                Tokens['meter_id'].append(result[0][1])
                print(result[0][1])

    global tkens
    tkens = Tokens
    print(itemList)
    print(Tokens)
    return Tokens

@app.get('/api/reading')
def getReading(img_path):
    coordinates = tkens['roi_coordinates'][0]
    url = img_path
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) # The image object
    # print(type(image))
    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    b1 = curr_datetime+'.jpg'
    # print(b1)
    cv2.imwrite(b1, image)
    image = 'C:/Users/sheri/bill-ocr-api/'+b1
    #read image
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # dictionary to store cropped coordinates
    itemList = {
      "image_path": img_path,
      "meter_reading": None,
    }

    # dictionary to store extracted content via OCR
    Tokens = {
        "meter_reading": [],
    }

    for item in itemList.keys():
      if item != "image_path":
        print("Select the coordinates for", item)

        #select ROI function
        scale_percent = 30
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img_res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #   roi = cv2.selectROI(img_res)

        #rectangle points of roi
        roi = coordinates
    #   print(roi)

        #Crop selected roi from raw image
        roi_cropped = img_res[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        scale_percent = 200
        width = int(roi_cropped.shape[1] * scale_percent / 100)
        height = int(roi_cropped.shape[0] * scale_percent / 100)
        dim = (width, height)
        img_res = cv2.resize(roi_cropped, dim, interpolation = cv2.INTER_AREA)
        #show cropped image
        # cv2.imshow(item, img_res)

        #save the cropped image
        itemList[item] = img_res

        # Get the current date and
        # time from system
        # and use strftime function
        # to format the date and time.
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

        # Split the picture path
        # into root and extension
        newfileName = item + ".jpg"
        splitted_path = os.path.splitext(newfileName)

        # Add the current date time
        # between root and extension
        modified_picture_path = splitted_path[0] + curr_datetime + splitted_path[1]

        cv2.imwrite(modified_picture_path, img_res)

        #hold window
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

        if item == 'meter_reading':
            # Initialize PaddleOCR
            ocr = PaddleOCR()

            # Perform OCR on the image
            result = ocr.ocr(img_res)

            print(result)
            output_text = ' '.join(item[1][0] for item in result[0])

            # Print the output text
            text = output_text
            Tokens['meter_reading'].append(text)
    print(itemList)
    print(Tokens)

    return Tokens

@app.get('/api/bill')
def getBill(url1, url2):
    urlA = url1
    urlB = url2
    urlist = [urlA, urlB]
    # dictionary to store extracted consumer, meter, account numbers via OCR
    Tokens = {
      "consumer_no": [],
      "meter_no": [],
      "account_no": []
    }
    
    counter = 0
    for i in urlist:
        # Initialize PaddleOCR
        ocr = PaddleOCR()
        
        #read image from url
        resp = urlopen(i)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR) # The image object
        print(type(image))
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        b1 = curr_datetime+'.jpg'
        print(b1)
        cv2.imwrite(b1, image)
        image_path = 'C:/Users/sheri/bill-ocr-api/'+b1
        
        # Provide the path to the image you want to extract text from
        img_path= image_path

        # Perform OCR on the image
        result = ocr.ocr(img_path)

#         print(result)
        output_text = ' '.join(item[1][0] for item in result[0])

        # Print the output text
        text = output_text
#         print(text)
#         print(type(text))

        # Split the text into tokens based on whitespace
        tokens = text.split()
        print(tokens)
        counter +=1
        
        if counter == 1:
            #account no
            account = tokens.index('AccountNumber')
            account_no = tokens[account+2]
            Tokens['account_no'].append(account_no)
        else:
            #consumer no
            consumer = tokens.index('10')
            consumer_no = tokens[consumer-1]
            consumer_no = consumer_no[0:8]
            Tokens['consumer_no'].append(consumer_no)

            #meter no
            meter = tokens.index('10')
            meter_no = tokens[meter+1]
            meter_no = meter_no[8::]
            Tokens['meter_no'].append(meter_no)

    print(Tokens)
    return Tokens
