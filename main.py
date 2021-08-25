import cv2
import mediapipe as mp
import numpy as np

BG_COLOR = (255, 255, 255)
SEGMENTATION_MODEL = 0


def segment_person(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = selfie_segmentation.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    output_image = np.where(condition, image, bg_image)
    return output_image


def detect_edges(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 1.41)

    edge = cv2.Canny(img_blur, 25, 75)
    return edge


def get_transparent_background(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    alpha = image[:, :, 3]
    alpha[np.all(image[:, :, 0:3] == (0, 0, 0), 2)] = 0
    image[:, :, 3] = alpha
    return image


def overlay(background, foreground):
    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    overlay = cv2.add(foreground, background)
    return cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)


mp_selfie_segmentation = mp.solutions.selfie_segmentation

# cap = cv2.VideoCapture(0)  # uncomment to test with webcam
cap = cv2.VideoCapture('path/to/video')
w = int(cap.get(3))
h = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('neon-twin.avi',
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      30, (w, h))

with mp_selfie_segmentation.SelfieSegmentation(model_selection=SEGMENTATION_MODEL) as selfie_segmentation:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        res = segment_person(frame)
        res = detect_edges(res)
        res = cv2.flip(res, 1)
        res = get_transparent_background(res)
        res = overlay(frame, res)

        out.write(res)
        cv2.imshow('Neon Twin', res)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
out.release()
print("Video Saved!")
