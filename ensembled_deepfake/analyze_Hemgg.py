import cv2, os, torch, numpy as np
# importante saber que la ultima version de transformers da error en este momento, "transformers==4.45.2" es la mejor que funciona actualmente
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# Ruta del modelo local

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_video_Hemgg(frames, fps, processor, model, interval: int, ia_threshold: float = 0.85):

    """
    interval: cantidad de frames saltados por cada frame analizado
    ia_threshold: a partir de que probabilidad se considera IA seguro
    """
    
    
    
    # Carga del modelo y preprocesador
    

    # datos para el proceso
    clip_length = 16 #constante para cada modelo
    clip_frames = []
    clip_count = 0
    is_ia = False
    ia_clips = 0
    predictions = []
    ia_predictions = [] # nivel de confianza de IA


    for i in range(0, len(frames), clip_length):
        clip_frames = frames[i:i+clip_length]
        print("hemgg")
        if clip_length == len(clip_frames):
            probs = process_video_hemgg(clip_frames, processor, model) 
                    
            # APLICACION DE LOGICA (si un clip es IA, todo el video se considera engañoso)
            ia_prob = probs[0][1].item()
            if ia_prob >= 0.5: 
                if ia_prob > ia_threshold:
                    is_ia = True
                ia_clips += 1
                ia_predictions.append(ia_prob) # nos servirá para calcular la confianza de la prediccion más adelante
            
            # Calcular segundo de inicio y fin del clip
            clip_start = round((i+1) * interval / fps, 2)
            clip_end = round((i+1+clip_length) * interval / fps, 2)

            # Almacenar resultados en una lista aparte para los clips
            predictions.append({
                "clip_num": clip_count+1,
                "clip_period": (clip_start, clip_end),
                "prediction": "IA" if ia_prob > 0.5 else "REAL",
                "confidence": round(ia_prob * 100, 2) #[0-100]
            })
            clip_count += 1      
        
        elif (clip_length // 2) <= len(clip_frames) < clip_length:
            
            # Calcular segundo de inicio y fin del clip
            clip_start = round((i+1) * interval / fps, 2)
            clip_end = round((i+1+len(clip_frames)) * interval / fps, 2)
            
            while len(clip_frames) < clip_length:
                clip_frames.append(frames[-1])
            probs = process_video_hemgg(clip_frames, processor, model)
                    
            # APLICACION DE LOGICA (si un clip es IA, todo el video se considera engañoso)
            ia_prob = probs[0][1].item()
            if ia_prob >= 0.5: 
                if ia_prob > ia_threshold:
                    is_ia = True
                ia_clips += 1
                ia_predictions.append(ia_prob) # nos servirá para calcular la confianza de la prediccion más adelante

            # Almacenar resultados en una lista aparte para los clips
            predictions.append({
                "clip_num": clip_count+1,
                "clip_period": (clip_start, clip_end),
                "prediction": "IA" if ia_prob > 0.5 else "REAL",
                "confidence": round(ia_prob * 100, 2) #[0-100]
            })


    # VIDEO PROCESADO 


    # Confianza media de los clips detectados como IA
    ia_confidences = np.mean(ia_predictions) if ia_predictions else 0

    

    # Resultado
    
    return {
        "label": "IA" if is_ia else "Real",
        "ia_confidence": ia_confidences, # [0-1]
        "ia_clips": ia_clips,
        "clips_info": predictions,
    }


def process_video_hemgg(clip_frames, processor, model):
    inputs = processor(images=clip_frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs