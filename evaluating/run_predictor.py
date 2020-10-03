from evaluating.predictor import ImagePredictor
import cv2
import matplotlib.pyplot as plt

weights_path = "architecture/weights/AlzheimerCNN_AlzheimerMPRageNoDeep_mobilenet_weights.h5"
model = ImagePredictor(weights_path)

image = cv2.imread("data/processed/alzheimer_mprage_nodeep/test/AD/011_S_0010 MPRAGE I91071 SLICE_80.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

image.shape

model.predict(image)

