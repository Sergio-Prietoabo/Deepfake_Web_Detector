import cv2
import torch
import numpy as np

from .deep_fake_detection.processor_deepfake import DeepFakeProcessor #hay que tener en cuenta que el modelo usa face-recognition, pero se ha eliminado del modelo la secuencia que lo requiere (puede cambiar el comportamiento)
from .deep_fake_detection.modeling import load_model

def analyze_video_Naman(frames, fps, processor, model, interval: int, ia_threshold: float = 0.85, device='cpu'):
    
    """
    Lógica de decisión: si tan solo un clip ha sido clasificado como IA, se considera que todo el video es engañoso 
    interval: cantidad de frames saltados por cada frame analizado
    ia_threshold: porcentaje minimo para clasificar el clip como IA
    device: cpu o cuda, para poder ejecutar el modelo usando cpu o gpu
    """

    

    # VARIABLES IMPORTANTES PARA EL MODELO
    clip_length = 20
    clip_frames = []
    clip_count = 0
    is_ia = False
    ia_clips = 0
    predictions = [] # informacion especifica de cada clip (util para almacenar en una mongo)
    ia_predictions = [] # nivel de confianza de IA


    # PROCESANDO VIDEO
    # El bucle a continuacion extrae bloques de 20 frames mediante read -> append, luego los procesa, guarda los resultados y empieza de nuevo hasta que el video termine.
    for i in range(0, len(frames), clip_length):
        clip_frames = frames[i:i+clip_length]
        print("naman")
        if clip_length == len(clip_frames):
            probs, pred_idx, confidence = process_video_naman(clip_frames, processor, model, device) 
                    
            # APLICACION DE LOGICA (si un clip es IA, todo el video se considera engañoso)
            ia_prob = probs[1]
            if ia_prob >= 0.5: 
                if ia_prob > ia_threshold:
                    is_ia = True
                ia_clips += 1
                ia_predictions.append(probs[1]) # nos servirá para calcular la confianza de la prediccion más adelante
            
            # Calcular segundo de inicio y fin del clip
            clip_start = round((i+1) * interval / fps, 2)
            clip_end = round((i+1+clip_length) * interval / fps, 2)

            # Almacenar resultados en una lista aparte para los clips
            predictions.append({
                "clip_num": clip_count+1,
                "clip_period": (clip_start, clip_end),
                "prediction": "IA" if pred_idx else "REAL",
                "confidence": round(confidence, 2) #[0-100]
            })
            clip_count += 1      
        
        elif (clip_length // 2) <= len(clip_frames) < clip_length:
            
            # Calcular segundo de inicio y fin del clip
            clip_start = round((i+1) * interval / fps, 2)
            clip_end = round((i+1+len(clip_frames)) * interval / fps, 2)
            
            while len(clip_frames) < clip_length:
                clip_frames.append(frames[-1])
            probs, pred_idx, confidence = process_video_naman(clip_frames, processor, model, device) 
                    
            # APLICACION DE LOGICA (si un clip es IA, todo el video se considera engañoso)
            ia_prob = probs[1]
            if ia_prob >= 0.5: 
                if ia_prob > ia_threshold:
                    is_ia = True
                ia_clips += 1
                ia_predictions.append(probs[1]) # nos servirá para calcular la confianza de la prediccion más adelante

            # Almacenar resultados en una lista aparte para los clips
            predictions.append({
                "clip_num": clip_count+1,
                "clip_period": (clip_start, clip_end),
                "prediction": "IA" if pred_idx else "REAL",
                "confidence": round(confidence, 2) #[0-100]
            })     
        

    # Confianza media de los clips detectados como IA
    ia_confidences = np.mean(ia_predictions) if ia_predictions else 0


    # Resultado
    if len(predictions) == 0: # Evitar el error si el video estaba vacío
        return None

    return {
        "label": "IA" if is_ia else "Real",
        "ia_confidence": ia_confidences, # [0-1]
        "ia_clips": ia_clips,
        "clips_info": predictions,
    }



def process_video_naman(clip_frames, processor, model, device):
    inputs = processor(frames=clip_frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        _, logits = model(pixel_values)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0] # array normalizado [%real, %fake]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx]) * 100
    
    return probs, pred_idx, confidence