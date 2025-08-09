"""
Configuration validation utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .cli_utils import cli


class ConfigValidator:
    """Configuration file validator"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_config_file(self, config_path: str) -> Tuple[bool, List[str], List[str]]:
        """Validate configuration file"""
        self.errors = []
        self.warnings = []
        
        # Check if file exists
        path = Path(config_path)
        if not path.exists():
            self.errors.append(f"Configuration file not found: {config_path}")
            return False, self.errors, self.warnings
        
        # Try to parse YAML
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {e}")
            return False, self.errors, self.warnings
        except Exception as e:
            self.errors.append(f"Error reading config file: {e}")
            return False, self.errors, self.warnings
        
        # Validate configuration content
        if config is None:
            self.errors.append("Configuration file is empty")
            return False, self.errors, self.warnings
        
        # Perform specific validation based on config type
        if 'models' in config:
            self._validate_evaluation_config(config)
        elif 'target_data_models' in config:
            self._validate_pipeline_config(config)
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_evaluation_config(self, config: Dict[str, Any]):
        """Validate evaluation configuration"""
        required_fields = ['workspace', 'llamafactory_dir', 'models']
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
        
        # Validate paths
        if 'workspace' in config:
            workspace = Path(config['workspace'])
            if not workspace.exists():
                self.warnings.append(f"Workspace directory does not exist: {workspace}")
        
        if 'llamafactory_dir' in config:
            llamafactory_dir = Path(config['llamafactory_dir'])
            if not llamafactory_dir.exists():
                self.warnings.append(f"LLaMA-Factory directory does not exist: {llamafactory_dir}")
        
        # Validate model configurations
        if 'models' in config:
            for model_name, model_config in config['models'].items():
                if not isinstance(model_config, dict):
                    self.errors.append(f"Model {model_name} configuration must be a dictionary")
                    continue
                
                # Check required model fields
                required_model_fields = ['name', 'base_model', 'template']
                for field in required_model_fields:
                    if field not in model_config:
                        self.errors.append(f"Model {model_name} missing required field: {field}")
    
    def _validate_pipeline_config(self, config: Dict[str, Any]):
        """Validate pipeline configuration"""
        required_fields = ['target_data_models', 'evaluation_settings']
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
        
        # Validate evaluation settings
        if 'evaluation_settings' in config:
            eval_settings = config['evaluation_settings']
            if 'sft_model' not in eval_settings:
                self.errors.append("Missing sft_model in evaluation_settings")
            
            if 'methods' not in eval_settings:
                self.errors.append("Missing methods in evaluation_settings")


def validate_and_load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Validate and load configuration file with user-friendly output"""
    if not config_path:
        cli.print_warning("No configuration file specified, using defaults")
        return None
    
    cli.print_info(f"Validating configuration: {config_path}")
    
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate_config_file(config_path)
    
    # Display warnings
    for warning in warnings:
        cli.print_warning(warning)
    
    # Display errors and exit if invalid
    if not is_valid:
        cli.print_error("Configuration validation failed!")
        for error in errors:
            cli.print_error(f"  • {error}")
        return None
    
    # Load the config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        cli.print_success(f"Configuration loaded successfully: {config_path}")
        return config
    except Exception as e:
        cli.print_error(f"Failed to load configuration: {e}")
        return None