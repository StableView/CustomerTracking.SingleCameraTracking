import cv2
import numpy as np

FONT_SCALE = 0.8
THICKNESS = 2
RED_COLOR = (0,0,255)
WHITE_COLOR = (255,255,255)
YELLOW_COLOR = (0, 255, 255)
POINT_TEXT1 = (5,20)
POINT_TEXT2 = (5,50)
POINT_ORIGIN_RECTANGLE = (0,0)

class GetPointsCameras:
    def __init__(self) -> None:
        self.points_reference: list = []
        self.points_camera: list = []
        self.image_reference: np.array = None
        self.image_camera:np.array = None
        self.image_concat:np.array = None
        self.image_height: int = 800 
        self.state: int = 0
        
    def get_points(self,image_reference: np.array,image_camera:np.array, win_name:str):
        self.image_reference = image_reference
        self.image_camera = image_camera
        self.image_concat = cv2.hconcat([self.image_reference,self.image_camera])
         
        h,w,_ = self.image_concat.shape
        r = h / w
        self.image_concat = cv2.resize(self.image_concat,(int(self.image_height/r),self.image_height))
        text = "Press p to save the points or ESC to exit."
        cv2.putText(self.image_concat, text, POINT_TEXT1, fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = FONT_SCALE, color = RED_COLOR, thickness=THICKNESS)
            
        cv2.namedWindow(winname = win_name) 
        cv2.setMouseCallback(win_name, self.capture_points) 
        while True:
            cv2.imshow(win_name, self.image_concat)       
            key = cv2.waitKey(1)
            if self.state == 1:
                self.state = 0

            if key == 27:
                self.points_camera = []
                self.points_reference = []
                cv2.destroyAllWindows()
                break
            if key == ord('p'):
                if self.state == 0 and len(self.points_reference)>=4:
                    self.state = 2
                elif self.state == 0 and len(self.points_reference)<4:
                    self.state = 1
                    text = "You must select at least 4 points,"
                    text2 = "press ENTER to continue or ESC to exit."
                    image_copy = self.image_concat.copy()
                    cv2.rectangle(image_copy,POINT_ORIGIN_RECTANGLE,(w,60),WHITE_COLOR,-1)
                    cv2.putText(image_copy, text, POINT_TEXT1, fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = FONT_SCALE, color = RED_COLOR, thickness=THICKNESS)
                    cv2.putText(image_copy, text2, POINT_TEXT2, fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = FONT_SCALE, color = RED_COLOR, thickness=THICKNESS)
                    cv2.imshow(win_name, image_copy)
                    key = cv2.waitKey(0)
                    if key == 27:
                        self.points_camera = []
                        self.points_reference = []
                        cv2.destroyAllWindows()
                        break
                elif (self.state == 2) & (len(self.points_camera)!=len(self.points_reference)):
                    self.state = 2
                    text = "The number of points selected in both images must match,"
                    text2 = "press ENTER to continue or ESC to exit."
                    image_copy = self.image_concat.copy()
                    cv2.rectangle(image_copy,POINT_ORIGIN_RECTANGLE,(w,60),WHITE_COLOR,-1)
                    cv2.putText(image_copy, text, POINT_TEXT1, fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = FONT_SCALE, color = RED_COLOR, thickness=THICKNESS)
                    cv2.putText(image_copy, text2, POINT_TEXT2, fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = FONT_SCALE, color = RED_COLOR, thickness=THICKNESS)
                    cv2.imshow(win_name, image_copy)
                    key = cv2.waitKey(0)
                    if key == 27:
                        self.points_camera = []
                        self.points_reference = []
                        cv2.destroyAllWindows()
                        break
                elif self.state == 1:
                    self.state = 0
                else :
                    self.state = 0
                    cv2.destroyAllWindows()
                    break

        points_reference = np.zeros((len(self.points_reference), 2), dtype=np.float32)
        points_camera = np.zeros((len(self.points_camera), 2), dtype=np.float32)
        for i in range(len(self.points_reference)):
            points_reference[i, :] = np.array(self.points_reference[i])
            points_camera[i, :] = np.array(self.points_camera[i])
        
        return points_reference,points_camera

    def capture_points(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONUP:
            h_reference,w_reference,_ = self.image_reference.shape
            h_camera,w_camera,_ = self.image_camera.shape
            h_concat,w_concat,_ = self.image_concat.shape
            r = w_concat/(w_reference+w_camera)
            self.x = int(x / r)
            self.y = int(y / r)
            
            if (self.state == 0) & (self.x <= w_reference):
                radius =5
                text = str(len(self.points_reference))
                cv2.putText(self.image_concat, text, (x,y), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = FONT_SCALE, color = RED_COLOR)
                cv2.circle(self.image_concat, (x, y), radius,YELLOW_COLOR,-1) 
                self.points_reference.append((self.x,self.y))
            if (self.state == 2) & (self.x > w_camera) & (len(self.points_camera) < len(self.points_reference)):
                radius =5
                text = str(len(self.points_camera))
                cv2.putText(self.image_concat, text, (x,y), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = FONT_SCALE, color = RED_COLOR)
                cv2.circle(self.image_concat, (x, y), radius,YELLOW_COLOR,-1) 
                self.points_camera.append((self.x-w_camera,self.y))
