import cv2, os
import torch
import numpy as np



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_video_jayala(frames, fps, model, interval: int, ia_threshold: float = 0.85, device='cpu'):
    
    # 1. Crear el modelo con los mismos parámetros que se entrenó
    

    # VARIABLES IMPORTANTES PARA EL MODELO
    clip_length = 30
    clip_frames = []
    clip_count = 0
    is_ia = False
    ia_clips = 0
    predictions = [] # informacion especifica de cada clip (util para almacenar en una mongo)
    ia_predictions = [] # nivel de confianza de IA


    # PROCESANDO VIDEO
    
    for i in range(0, len(frames), clip_length):
        clip_frames = frames[i:i+clip_length]
        print("jayala")
        
        if clip_length == len(clip_frames):
            # Calcular segundo de inicio y fin del clip
            clip_start = round((i+1) * interval / fps, 2)
            clip_end = round((i+1+clip_length) * interval / fps, 2)
            
            pred_idx, confidence = process_video_jayala(clip_frames, model, device)
                    
            # APLICACION DE LOGICA (si un clip es IA, todo el video se considera engañoso)
            if pred_idx:
                if confidence >= ia_threshold:
                    is_ia = True
                ia_clips += 1
                ia_predictions.append(confidence) # nos servirá para calcular la confianza con la que se ha detectado IA en la prediccion más adelante


            # Almacenar resultados en una lista aparte para los clips
            predictions.append({
                "clip_num": clip_count+1,
                "clip_period": (clip_start, clip_end),
                "prediction": "IA" if pred_idx else "REAL",
                "confidence": round(confidence*100, 2) # [0-100]
            })
            clip_count += 1    
        
        elif (clip_length // 2) <= len(clip_frames) < clip_length:
            
            # Calcular segundo de inicio y fin del clip
            clip_start = round((i+1) * interval / fps, 2)
            clip_end = round((i+1+len(clip_frames)) * interval / fps, 2)
            
            while len(clip_frames) < clip_length:
                clip_frames.append(frames[-1])
            pred_idx, confidence = process_video_jayala(clip_frames, model, device)
                    
            # APLICACION DE LOGICA (si un clip es IA, todo el video se considera engañoso)
            if pred_idx:
                if confidence >= ia_threshold:
                    is_ia = True
                ia_clips += 1
                ia_predictions.append(confidence) # nos servirá para calcular la confianza con la que se ha detectado IA en la prediccion más adelante


            # Almacenar resultados en una lista aparte para los clips
            predictions.append({
                "clip_num": clip_count+1,
                "clip_period": (clip_start, clip_end),
                "prediction": "IA" if pred_idx else "REAL",
                "confidence": round(confidence*100, 2) # [0-100]
            })
   
    # VIDEO PROCESADO 


    # Confianza media de los clips detectados como IA
    ia_confidences = np.mean(ia_predictions) if ia_predictions else 0


    return {
        "label": "IA" if is_ia else "Real",
        "ia_confidence": ia_confidences, # [0-1]
        "ia_clips": ia_clips,
        "clips_info": predictions,
    }



def process_video_jayala(frames, model, device): 
    """
    Este modelo trabaja frame a frame a diferencia de los otros, pero vamos a dividir el video en clips de 20 frames y vamos a trabajar cada bloque por separado para que los resultados sean comparables entre modelos
    """
    
    clip_frames = preprocess_frames(frames, device)
    ia_frames = []
    all_probs = []
    batch = torch.stack(clip_frames)  # [30,3,224,224]
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)  # probs tiene forma [30][2], (30 arrays de forma [real, fake] uno por cada frame)


    # Aplicación de lógica
    fake_frames = 0
    for p in probs:
        fake_prob = p[1].item()
        all_probs.append(fake_prob)
        if fake_prob > 0.5:
            fake_frames += 1
            ia_frames.append(fake_prob) # Para poder hacer la media exclusiva de los frames clasificados como IA
    
    ia_confidence = np.mean(all_probs)
    pred_idx = 1 if ia_confidence > 0.5 else 0 # La media de la confianza de IA tiene que ser mayor a 0.5 o no se cuenta como IA
        

    
    if pred_idx: # Si el modelo detecta IA, se almacena exclusivamente la confianza de los clips detectados como IA, para ver con que confianza ha detectado la IA
        if fake_frames >= 15: # Si al menos 50% de los frames han sido detectados como IA, la confianza va a ser solamente de los clips IA (sin diluir)
            confidence = np.mean(ia_frames) 
        else:
            confidence = ia_confidence
    else:       # Si no detecta IA, se almacena la probabilidad media de todo el clip
        confidence = (1 - ia_confidence)

    return pred_idx, confidence.item()



def preprocess_frames(frames, device):
    clip_frames = []
    for frame in frames:
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # resize to 224x224
        frame = cv2.resize(frame, (224, 224))

        # numpy → tensor float32
        frame = torch.from_numpy(frame).float() / 255.0  # [224,224,3]

        # Normalize (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # HWC → CHW
        frame = frame.permute(2, 0, 1)  # [3,224,224]
        
        # Apply normalization
        frame = (frame - mean) / std

        # move to device
        frame = frame.to(device)
        
        clip_frames.append(frame)

    return clip_frames