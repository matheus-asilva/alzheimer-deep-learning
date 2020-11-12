import gradio as gr
import nibabel as nib
import importlib
from tensorflow.keras.preprocessing import image as image_fn
import matplotlib.pyplot as plt
import numpy as np

predictor_module = importlib.import_module("evaluating.predictor")
ImagePredictor = getattr(predictor_module, "ImagePredictor")

# Carrega pesos da CNN importa o modelo
weights_path = "architecture/weights/AlzheimerCNN_AlzheimerMPRageNoDeep_mobilenet_weights_sigmoid.h5"
model = ImagePredictor(weights_path)

def nii_predict(nii_file):

    nifti_map = nib.Nifti1Image.make_file_map()
    nifti_map['image'].fileobj = nii_file
    
    nii = nib.Nifti1Image.from_file_map(nifti_map).get_fdata()
    
    # Define escala de cores para cinza
    colormap = plt.cm.get_cmap('gray')

    # Lista com os predicts
    preds = []

    # Faz predict do slice 80~110
    for i in range(80, 111):

        # Seleciona slice a partir da visão coronal e rotaciona -90 graus a imagem
        img = np.rot90(np.squeeze(nii[:, i, :]))

        # Mapeia a escala de cores
        normed_img = (img - np.min(img)) / (np.max(img) - np.min(img))
        mapped_img = colormap(normed_img)

        # Faz reshape para 224 x 224
        reshaped_img = np.resize(mapped_img, (1, 224, 224, 3))

        # Predict na matriz e salva na lista
        preds.append(model.predict(reshaped_img)[0])

    # Define classe: {0: "Saudável", 1: "Alzheimer"}
    classes = [np.argmax(p) for p in preds]
    predominant_class = max(set(classes), key=classes.count)

    # Cria dicionário com melhores predições
    preds_matrix = np.stack(preds)
    best_pred = preds_matrix[:, predominant_class].argmax()

    mapping = {0: "Saudável", 1: "Alzheimer"}

    d = {mapping[i]: k for i, k in enumerate(preds_matrix[best_pred, :], 0)}
    title = "\n" + ('\n'.join([f'{key}: {value}' for key, value in d.items()]))
    
    plt.imshow(mapped_img)
    plt.title(title)

    return plt


gr.Interface(
    fn=nii_predict,
    inputs="file",
    outputs="plot",
    title="Alzheimer Prediction Interface with Convolutional Neural Networks"
).launch(share=False)
