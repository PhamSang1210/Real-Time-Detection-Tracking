import os
import sys
sys.path.insert(0, './detection_module')
from detection_module.models.common import DetectMultiBackend
from detection_module.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from detection_module.utils.plots import Annotator, colors
from tracking_module.utils.parser import get_config
from tracking_module.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import streamlit as st
from google_drive_downloader import GoogleDriveDownloader as gdd

print(torch.cuda.is_available())  # This should return True if supported
print(torch.version.cuda)  

# ADD pencirl
class_label = ["nguoi", "xe dap", "o to", "xe may", "may bay", "xe buyt", "tau hoa", "xe tai", "thuyen", "den giao thong",
               "voi chua chay", "bien bao dung", "dong ho do xe", "bang ghe", "chim", "meo", "cho", "ngua", "cuu", "bo",
               "voi", "gau", "ngua van", "huou cao co", "ba lo", "o", "tui xach", "ca vat", "vali", "dia bay",
               "van truot", "van truot tuyet", "bong the thao", "dieu", "gay bong chay", "gang tay bong chay", "van truot", "van luot song",
               "vot tennis", "chai", "ly ruou", "coc", "nia", "dao", "muong", "bat", "chuoi", "tao",
               "sandwich", "cam", "sup lo xanh", "ca rot", "xuc xich", "pizza", "banh ran", "banh ngot", "ghe", "di vang",
               "chau cay", "giuong", "ban an", "toilet", "tv", "may tinh xach tay", "chuot", "dieu khien tu xa", "ban phim", "dien thoai di dong",
               "lo vi song", "lo nuong", "may nuong banh mi", "bon rua", "tu lanh", "sach", "dong ho", "binh hoa", "keo", "gau bong",
               'may say toc', 'ban chai danh rang', "but"]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def predict(model, deepsort_model, img):
    img_org = img.copy()
    h, w, s = img_org.shape
    img = letterbox(img, new_shape=640)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)
    image = img.astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, 0)

    image = torch.from_numpy(image)

    pred = model(image)

    # no choose class is all class
    pred = non_max_suppression(pred, 0.5, 0.5)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        annotator = Annotator(img_org, line_width=2, pil=not ascii)
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                image.shape[2:], det[:, :4], img_org.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            # pass detections to deepsort
            outputs = deepsort_model.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img_org)

            # draw boxes for visualization
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]

                    c = int(cls)  # integer class
                    label = f"{id} {class_label[c]} {round(float(conf), 2)}"
                    annotator.box_label(bboxes, label, color=colors(c, True))

        # Stream results
        img_org = annotator.result()

    return img_org


if __name__ == "__main__":
    st.header("✨SANG - Real-Time Detection & Tracking")

    if not os.path.exists('./yolov5n.pt'):
        with st.spinner(text="Download detection model in progress..."):
            gdd.download_file_from_google_drive(file_id='1V5hUspqnI6uvBIPyccga9lsz8-fWFQ9p',
                                                dest_path='./yolov5n.pt')

    if not os.path.exists('./ckpt.t7'):
        with st.spinner(text="Download tracking model in progress..."):
            gdd.download_file_from_google_drive(file_id='1GJpFNw0fU-6X1z8_x_Mb7f5A9pRtLZt_',
                                                dest_path='./ckpt.t7')

    if not os.path.exists('./crowdhuman_yolov5m.pt'):
        with st.spinner(text="Download tracking model in progress..."):
            gdd.download_file_from_google_drive(file_id='1Bz_tZia6BeAy7PW1LJm5x8469D0ooDtQ',
                                                dest_path='./crowdhuman_yolov5m.pt')

    deepsort = DeepSort(model_path='ckpt.t7', use_cuda=True)
    model = DetectMultiBackend(weights='model.pt', device='cpu')
    # model = DetectMultiBackend(weights='model.pt', device='cuda')


    # Sử dụng webcam
    cap = cv2.VideoCapture(0)  # 0 là thiết bị webcam mặc định
    stframe = st.empty()
    frame_delay = 0.01

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận khung hình. Kết thúc...")
            break
        frame = predict(model, deepsort, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame)
        time.sleep(frame_delay)  # Điều chỉnh tốc độ hiển thị

    cap.release()
    cv2.destroyAllWindows()
