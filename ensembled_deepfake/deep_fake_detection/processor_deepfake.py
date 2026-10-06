import numpy as np
import torch
from PIL import Image
import cv2
# import face_recognition  se ha comentado esto para evitar sesgos
from transformers import ProcessorMixin

class DeepFakeProcessor(ProcessorMixin):
    """Processor for DeepFake detection model."""
    
    def __init__(self, im_size=112, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.im_size = im_size
        self.mean = mean
        self.std = std
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame.
        
        Args:
            frame: PIL Image or numpy array
            
        Returns:
            torch.Tensor: Processed frame tensor
        """
        # Convert to PIL Image if it's a numpy array
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        
        # Resize
        frame = frame.resize((self.im_size, self.im_size))
        
        # Convert to tensor
        frame = np.array(frame).astype(np.float32) / 255.0
        frame = (frame - np.array(self.mean)) / np.array(self.std)
        frame = frame.transpose(2, 0, 1)  # HWC -> CHW
        frame = torch.tensor(frame, dtype=torch.float32)
        
        return frame
    
    def extract_frames(self, video_path, sequence_length=20):
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            sequence_length: Number of frames to extract
            extract_faces: Whether to extract faces from frames
            
        Returns:
            torch.Tensor: Tensor of shape (1, sequence_length, 3, im_size, im_size)
        """
        frames = []
        
        # Open video file
        vidObj = cv2.VideoCapture(video_path)
        
        # Calculate frame interval
        total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // sequence_length)
        
        # Extract frames
        count = 0
        success = True
        
        while success and len(frames) < sequence_length:
            success, image = vidObj.read()
            
            if success and count % interval == 0:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                """
                # Extract face if requested
                if extract_faces:
                    face_locations = face_recognition.face_locations(image)
                    if face_locations:
                        top, right, bottom, left = face_locations[0]
                        # Add padding
                        padding = 40
                        h, w = image.shape[:2]
                        top = max(0, top - padding)
                        bottom = min(h, bottom + padding)
                        left = max(0, left - padding)
                        right = min(w, right + padding)
                        image = image[top:bottom, left:right]
                """
                # Preprocess frame
                processed_frame = self.preprocess_frame(image)
                frames.append(processed_frame)
                
            
            count += 1
        
        # If we couldn't extract enough frames, duplicate the last one
        while len(frames) < sequence_length:
            frames.append(frames[-1] if frames else torch.zeros((3, self.im_size, self.im_size)))
        
        # Stack frames
        frames = torch.stack(frames)
        
        # Add batch dimension
        frames = frames.unsqueeze(0)
        
        return frames
    
    def __call__(self, video_path=None, frames=None, return_tensors="pt", **kwargs):
        """
        Process video for the model.
        
        Args:
            video_path: Path to the video file
            frames: List of frames (PIL Images or numpy arrays)
            return_tensors: Return format (only "pt" supported)
            
        Returns:
            dict: Processed inputs for the model
        """
        if return_tensors != "pt":
            raise ValueError("Only 'pt' return tensors are supported")
        
        if video_path is not None:
            # Extract frames from video
            sequence_length = kwargs.get("sequence_length", 20)
            extract_faces = kwargs.get("extract_faces", True)
            processed_frames = self.extract_frames(
                video_path, 
                sequence_length=sequence_length,
                extract_faces=extract_faces
            )
        elif frames is not None:
            # Process provided frames
            processed_frames = torch.stack([self.preprocess_frame(frame) for frame in frames])
            processed_frames = processed_frames.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError("Either video_path or frames must be provided")
        
        return {"pixel_values": processed_frames}
