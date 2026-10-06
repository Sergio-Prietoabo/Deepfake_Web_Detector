import torch
import torch.nn as nn
import torchvision.models as models
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class DeepFakeDetectorConfig:
    """Configuration class for DeepFakeDetector."""
    
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, 
                 bidirectional=False, sequence_length=20, im_size=112, **kwargs):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.im_size = im_size
        
    @classmethod
    def from_dict(cls, config_dict):
        """Create a configuration from a dictionary."""
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert configuration to a dictionary."""
        return {
            "num_classes": self.num_classes,
            "latent_dim": self.latent_dim,
            "lstm_layers": self.lstm_layers,
            "hidden_dim": self.hidden_dim,
            "bidirectional": self.bidirectional,
            "sequence_length": self.sequence_length,
            "im_size": self.im_size
        }


class DeepFakeDetectorModel(PreTrainedModel):
    """DeepFake detection model using ResNext50 and LSTM."""
    
    config_class = DeepFakeDetectorConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.num_classes
        self.latent_dim = config.latent_dim
        self.lstm_layers = config.lstm_layers
        self.hidden_dim = config.hidden_dim
        self.bidirectional = config.bidirectional
        
        # Initialize ResNext50 backbone
        resnext = models.resnext50_32x4d(pretrained=True)
        self.backbone = nn.Sequential(*list(resnext.children())[:-2])
        
        # Initialize LSTM
        self.lstm = nn.LSTM(
            self.latent_dim, 
            self.hidden_dim,
            self.lstm_layers, 
            bidirectional=self.bidirectional
        )
        
        # Additional layers
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x, labels=None):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            labels: Optional labels for computing loss
            
        Returns:
            SequenceClassifierOutput: Model outputs including loss and logits
        """
        batch_size, seq_length, c, h, w = x.shape
        
        # Reshape for ResNext processing
        x = x.view(batch_size * seq_length, c, h, w)
        
        # Extract features using ResNext
        features = self.backbone(x)
        
        # Apply average pooling
        pooled = self.avgpool(features)
        
        # Reshape for LSTM processing
        pooled = pooled.view(batch_size, seq_length, self.latent_dim)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(pooled, None)
        
        # Get the final time step output
        final = lstm_out[:, -1, :]
        
        # Apply dropout and classification
        logits = self.classifier(self.dropout(final))
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a pretrained model."""
        # Load config
        config_dict = kwargs.pop("config", None)
        if config_dict is None:
            config_dict = {
                "num_classes": 2,
                "latent_dim": 2048,
                "lstm_layers": 1,
                "hidden_dim": 2048,
                "bidirectional": False,
                "sequence_length": 20,
                "im_size": 112
            }
        
        config = DeepFakeDetectorConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config, *model_args, **kwargs)
        
        # Load weights
        state_dict = torch.load(pretrained_model_name_or_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model
