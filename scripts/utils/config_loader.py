import yaml
import json
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """model related configuration"""
    model_name: str
    batch_size: int
    learning_rate: float
    # add other model parameters...

@dataclass
class DataConfig:
    """data related configuration"""
    data_path: str
    train_ratio: float
    val_ratio: float
    # add other data parameters...

@dataclass
class Config:
    """total configuration class"""
    model: ModelConfig
    data: DataConfig
    # add other configuration classes...

class ConfigLoader:
    """configuration loader"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        load configuration file/load json file
        Args:
            config_path: configuration file path
        Returns:
            configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_ext == '.yaml' or file_ext == '.yml':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif file_ext == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")
            return config
        except Exception as e:
            raise Exception(f"Error loading config file: {str(e)}")
    


