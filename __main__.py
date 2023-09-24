import cv2

from src.detection import Yolov8, Draw


yolo = Yolov8('ObjectDetection/model/yolov8n.pt')

video_path = "videos/cctv_retail.mp4"

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('videos/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


while cap.isOpened():

    success, frame = cap.read()

    if success:
        
        results = yolo.predict_to_BoundingBoxes(frame)
        
        annotated_frame = Draw.plot_boxes(results)

        out.write(annotated_frame)
        #cv2.imshow("YOLOv8 Inference", annotated_frame)

        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break
    else:
        break

cap.release()
out.release()
#cv2.destroyAllWindows()
