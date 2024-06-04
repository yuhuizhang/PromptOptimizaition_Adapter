# written by Ruilin Cai 

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from yival.schemas.varation_generator_configs import BaseVariationGeneratorConfig

@dataclass
class AzureVariationGeneratorConfig(BaseVariationGeneratorConfig):
    
    model_name: str = os.getenv("model"),
    variations: Optional[List[str]] = None  # List of variations to generate
    diversify: bool = False,
    max_tokens: int = int(os.getenv("max_tokens")),
    variables: Optional[List[str]] = None
    prompt: Union[str, List[Dict[str, str]]] = "",