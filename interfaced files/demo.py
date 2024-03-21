#command to interface model
#python3 C:\avani\TRP\PaddleOCR\tools\export_model.py -c C:\avani\TRP\PaddleOCR\configs\det\det_r50_vd_sast_icdar15.yml -o Global.pretrained_model=C:\avani\TRP\PaddleOCR\det_r50_vd_sast_icdar15_v2.0_train\best_accuracy Global.save_inference_dir=C:\avani\TRP\PaddleOCR\inference\det_sast_ic15


from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image
import numpy as np

# Initialize PaddleOCR
det_algo={
    "east":"/PaddleOCR/inference/det_r50_east",
    "sast":"/PaddleOCR/inference/det_sast_ic15",
    "pse":"/PaddleOCR/inference/det_pse",
    "fce":"/PaddleOCR/inference/det_fce"
}
print("Choose an option:")
for i, option in enumerate(det_algo.keys(), start=1):
    print(f"{i}. {option}")
# Get user input
selected_option_index_det = int(input("Enter the number corresponding to your choice: ")) - 1

# Get the selected key
selected_key_det = list(det_algo.keys())[selected_option_index_det]
selected_value_det = det_algo[selected_key_det]

# Output the selected key-value pair
rec_algo={
    "srn":"/PaddleOCR/inference/rec_srn",
    "nrtr":"/PaddleOCR/inference/rec_mtb_nrtr",
    "svtr" :"/PaddleOCR/inference/rec_svtr_tiny_stn_en",
    "abi" :"/PaddleOCR/inference/rec_r45_abinet"}

print("Choose an option:")
for i, option in enumerate(rec_algo.keys(), start=1):
    print(f"{i}. {option}")
# Get user input
selected_option_index_rec = int(input("Enter the number corresponding to your choice: ")) - 1

# Get the selected key
selected_key_rec = list(rec_algo.keys())[selected_option_index_rec]
selected_value_rec = rec_algo[selected_key_rec]

ocr = PaddleOCR(det_model_dir=f"{selected_value_det}", rec_model_dir=f"{selected_value_rec}", use_angle_cls=True, use_gpu=False, rec_char_type='en')

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
