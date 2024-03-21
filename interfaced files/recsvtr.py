#command to interface model
#python3 tools/export_model.py -c configs/rec/rec_svtrnet.yml -o Global.pretrained_model=./rec_svtr_tiny_none_ctc_en_train/best_accuracy  Global.save_inference_dir=./inference/rec_svtr_tiny_stn_en

from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image
import numpy as np

# Initialize PaddleOCR
ocr = PaddleOCR(det_model_dir=None, rec_model_dir=r"/PaddleOCR/inference/rec_svtr_tiny_stn_en", use_angle_cls=True, use_gpu=False, gpu_id=0, rec_char_type='en',rec_image_shape="1,64,256")

# Open a connection to the webcam (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Perform OCR on the captured frame
    result = ocr.ocr(frame, cls=True)

    # Check if OCR result is not None and has elements
    if result and len(result) > 0 and result[0] is not None:
        result = result[0]  # Assuming you are interested in the first result
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path=r'C:\avani\TRP\PaddleOCR\doc\fonts\simfang.ttf')
        frame_with_ocr = cv2.cvtColor(np.array(im_show), cv2.COLOR_RGB2BGR)

        # Display the frame with OCR result
        cv2.imshow('Live Text Recognition', frame_with_ocr)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

