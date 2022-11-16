import cv2
import numpy as np
import pandas as pd
from follower import Follower


class VideoAnalyzer:
    def __init__(self) -> None:
        self.ball_track = pd.DataFrame(columns=['id', 'time', 'x', 'y', 'speed', 'event'])
        self.follow = Follower()
        self.detections = []

    def run(self, video: str):
        #Reading the video
        vidcap = cv2.VideoCapture(video)
        success,image = vidcap.read()
        count = 0
        success = True
        idx = 0
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))

        size = (frame_width, frame_height)
        result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, size)
        #Read the video frame by frame
        while success:
            #result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, size)
            #converting into hsv image
            hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            #hsv = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #green range
            lower_green = np.array([40,40, 40])
            upper_green = np.array([70, 255, 255])
            #blue range
            lower_blue = np.array([110,50,50])
            upper_blue = np.array([130,255,255])

            #Red range
            lower_red = np.array([0,31,255])
            upper_red = np.array([176,255,255])

            #white range
            lower_white = np.array([0,0,200], dtype=np.uint8)
            upper_white = np.array([255,255,255], dtype=np.uint8)

            #Define a mask ranging from lower to uppper
            mask = cv2.inRange(hsv, lower_green, upper_green)
            #Do masking
            res = cv2.bitwise_and(image, image, mask=mask)
            #convert to hsv to gray
            res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
            res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

            #Defining a kernel to do morphological operation in threshold image to 
            #get better output.
            kernel = np.ones((13,13),np.uint8)
            thresh = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            #find contours in threshold image     
            #im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            prev = 0
            font = cv2.FONT_HERSHEY_SIMPLEX

            for c in contours:
                x,y,w,h = cv2.boundingRect(c)

                #Detect players
                if(h>=(1.5)*w):
                    if(w>15 and h>= 15):
                        idx = idx+1
                        player_img = image[y:y+h,x:x+w]
                        player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
                        #If player has blue jersy
                        mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                        res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                        res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                        res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                        nzCount = cv2.countNonZero(res1)
                        #If player has red jersy
                        mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
                        res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                        res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
                        res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
                        nzCountred = cv2.countNonZero(res2)

                        if(nzCount >= 20):
                            #Mark blue jersy players as france
                            cv2.putText(image, 'WHITE TEAM', (x-2, y-2), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
                            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),3)
                        else:
                            pass
                        if(nzCountred>=20):
                            #Mark red jersy players as belgium
                            cv2.putText(image, 'RED TEAM', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
                            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
                        else:
                            pass
                if((h>=1 and w>=1) and (h<=30 and w<=30)):
                    player_img = image[y:y+h,x:x+w]

                    player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
                    #white ball  detection
                    mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
                    res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                    res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                    res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                    nzCount = cv2.countNonZero(res1)

                    if(nzCount >= 3):
                        # detect football
                        cv2.putText(image, 'football', (x-2, y-2), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
                        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)

                        self.detections.append([x, y, w, h]) #SE ALMACENA INFORMACIÓN DEL RECTANGULO

            info_id = self.follow.rastreo(self.detections) #Obtenemos la informacón de la pelota, mandando la detección de esta
            for inf in info_id:#Mostramos los datos recabados
                print(inf)
                x = (inf[0] + inf[2]) / 2
                y = (inf[1] + inf[3]) / 2

                values = {
                    'id': inf[4],
                    'time': 0,
                    'x': x,
                    'y': y,
                }
                self.ball_track.append(values, ignore_index=True)

            #cv2.imwrite("./videos/img/frame%d.jpg" % count, res)
            print('Read a new frame: ', success)     # save frame as JPEG file	
            count += 1
            cv2.imshow('Match Detection',image)
            result.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,image = vidcap.read()
            
        vidcap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    v = VideoAnalyzer()
    v.run('test_43.mp4')
