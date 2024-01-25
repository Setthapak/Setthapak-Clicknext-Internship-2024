from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2



# Load YOLO model
model = YOLO("yolov8n.pt")


def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        confidence = box.conf


        # Check if the detected object is a cat
        if model.names[int(class_id)] == "cat":
            coordinator = box.xyxy[0]
            class_name = "cat"

    # Draw bounding box
            annotator.box_label(
            box=coordinator, label=class_name, color=(255,0,0)
    )

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model(frame)

    for result in results:
        frame = draw_boxes(frame, result.boxes)
       
        

    return frame


if __name__ == "__main__":
    video_path = "D:\clicknext\internship-2024-main\CatZoomies.mp4"
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    # video_writer = cv2.VideoWriter(
    #     video_path + "D:\ClickNext_test\internship-2024-main\CatZoomies.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 60, (1280, 720)
    # )

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()

        if ret:
            # Detect cat from image frame
            frame_result = detect_object(frame)

            # Write result to video
            # video_writer.write(frame_result)

         # Show result
            cv2.namedWindow("CatZoomies_Detect", cv2.WINDOW_NORMAL)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (620, 50)
            fontScale = 1
            color = (0,0,255)
            thickness = 2
            cv2.putText(frame,'Setthapak + Clicknext-Internship-2024', org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
            
            cv2.imshow("CatZoomies_Detect", frame_result)
            if cv2.waitKey(1) & 0xFF == ord('q') :
                break

        else:
            break
   
    


    # Release the VideoCapture object and close the window
    cv2.video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
