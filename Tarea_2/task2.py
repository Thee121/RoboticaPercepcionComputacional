import os
import joblib
import numpy as np
import cv2
from pathlib import Path
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path

# Configuración HOG
HOG_PARAMS = {
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'orientations': 9,
    'block_norm': 'L2-Hys',
    'transform_sqrt': True
}
IMAGE_SIZE = (64, 64)

labels_dict = {'Right': 0, 'Left': 1, 'Forward': 2}
labels_inv = {v: k for k, v in labels_dict.items()}
data, labels = [], []

def segment_red_hsv(image):

    # Máscaras para rojo (intervalos de H para rojo puro)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0,100,50), (10,255,255))
    m2 = cv2.inRange(hsv, (160,100,50), (180,255,255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    return mask

def train_symbol_classifier():
    base_dir = Path.cwd()
    media_path = base_dir / 'Media' / 'Imagenes_Marcas' / 'Arrow'
    global data
    global labels    
    for name, lbl in labels_dict.items():
        folder = media_path / name        
        for fn in os.listdir(folder):
            img_path = folder / fn
            try:
                img = cv2.imread(img_path)
            except:
                continue

            mask = segment_red_hsv(img)
            # Buscar contornos
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < 300:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[y:y+h, x:x+w]

            if roi.size == 0:
                continue
            
            roi = cv2.resize(roi, IMAGE_SIZE)
            feat = hog(roi, **HOG_PARAMS)
            data.append(feat)
            labels.append(lbl)

    data = np.array(data)
    labels = np.array(labels)

    print("Entrenando KNN con HOG. Shape de data:", data.shape)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(data, labels)
    
    base_dir = Path.cwd()
    save_path = base_dir / 'Tarea_2' / 'symbol_knn_hog.pkl'
    joblib.dump({
        'knn': knn,
        'hog_params': HOG_PARAMS,
        'image_size': IMAGE_SIZE,
        'labels_inv': labels_inv
    }, save_path)

    print("Modelo HOG+KNN entrenado y guardado en symbol_knn_hog.pkl")
    return knn

def main():
    base_dir = Path.cwd()
    media_path = base_dir / 'Media'
    media_path_video = media_path / 'videos' / 'videoC.avi'

    if not os.path.exists("symbol_knn_hog.pkl"):
        knn_model = train_symbol_classifier()
    else:
        model_data = joblib.load("symbol_knn_hog.pkl")
        knn_model = model_data['knn']

    if not os.path.exists(media_path_video):
        print("No se encontró el video de entrada.")
        return

    print("Procesando video...")
    total_frames = 0
    flechas_detectadas = 0
    arrow_visible = False  # Track arrow visibility
    cap = cv2.VideoCapture(str(media_path_video))
    
    while(True):
        ret, frame = cap.read()
        
        if not ret:
            break

        mask = segment_red_hsv(frame)
        
        # Buscar contornos
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            arrow_visible = False
            total_frames += 1
            continue

        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 300:
            arrow_visible = False
            total_frames += 1
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+h, x:x+w]

        if roi.size == 0:
            arrow_visible = False
            total_frames += 1
            continue
        
        roi = cv2.resize(roi, IMAGE_SIZE)
        feat = hog(roi, **HOG_PARAMS)
        pred = knn_model.predict([feat])[0]
        
        direccion = labels_inv[pred]

        if direccion and not arrow_visible:
            print(f"Flecha detectada: {direccion}")
            flechas_detectadas += 1
            arrow_visible = True
        elif direccion is None:
            arrow_visible = False

        total_frames += 1

    print(f"Total de fotogramas procesados: {total_frames}")
    print(f"Flechas detectadas: {flechas_detectadas}")

if __name__ == "__main__":
    main()
