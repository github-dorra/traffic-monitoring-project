import os
import sys
import cv2
import torch
from ultralytics import YOLO
import numpy as np 
from speed import SpeedEstimator 



ACCIDENT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Accident-Detection-System')
if ACCIDENT_DIR not in sys.path:
    sys.path.append(ACCIDENT_DIR)
    
from detection import AccidentDetectionModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé: {device}")

model = YOLO("models/yolov10n.pt").to(device)
names = model.model.names 

ACCIDENT_MODEL_PATH = '../Accident-Detection-System/mon_classifieur_accident_complet.h5' 
accident_classifier = AccidentDetectionModel(ACCIDENT_MODEL_PATH)


# Configuration de l'estimateur de vitesse
line_pts = [(101, 371), (1035, 367)]
speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Coordonnées de la souris: ({x}, {y})")

# Chargement de la vidéo et configuration de la sortie
cap = cv2.VideoCapture('..\Accident-Detection-System\Demo.gif')
frame_width, frame_height = 1280, 720
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_annotated.mp4', fourcc, fps, (frame_width, frame_height))

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

frame_count = 0


while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin du flux vidéo ou échec de lecture.")
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))
    
    annotated_frame = frame.copy() 

    result = model.track(annotated_frame, persist=True, classes=[2, 7, 3, 5, 1, 6], verbose=False, device=device)

    if result and hasattr(result[0], "boxes") and result[0].boxes.id is not None:
        annotated_frame = speed_obj.estimate_speed(annotated_frame, result)
    
    # --- Étape B: Système d'Accident (Keras CNN) ---
    accident_alert = False
    cnn_confidence = 0.0
    
    if accident_classifier:
        input_for_cnn = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (250, 250))
        pred_label, pred_probs = accident_classifier.predict_accident(
            input_for_cnn[np.newaxis, :, :]
        )
        
        cnn_confidence = round(pred_probs[0][0] * 100, 2)
        cnn_sees_accident = (pred_label == "Accident")
        
        # Affichage du score CNN 
        color = (0, 255, 0) if not cnn_sees_accident else (0, 165, 255)
        cv2.putText(annotated_frame, f"Score CNN Accident: {cnn_confidence:.2f}%", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if cnn_sees_accident and cnn_confidence >= 30.5:
            accident_alert = True
    
    if accident_alert:
        cv2.putText(annotated_frame, "!!! ALERTE ACCIDENT DETECTEE !!!", (20, 60), 
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 4)
    else:
        cv2.putText(annotated_frame, "!!! NO ACCIDENT DETECTEE !!!", (20, 60), 
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 4)
        
    
    cv2.imshow("RGB", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Nettoyage
cap.release()
out.release()
cv2.destroyAllWindows()