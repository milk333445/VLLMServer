import yaml
import logging
import os

from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer

from embedding_engine.embed_rerank import Embedder, Reranker


class EmbedRerankBuilder:
    def __init__(self, config_path, logger=None):
        assert config_path.endswith(('yaml', 'yml')), "Config file must be a YAML file."
        self.config = self.load_config(config_path)
        
        self.embedding_model_configs = self.config.get('embedding_models', {})
        self.reranking_model_configs = self.config.get('reranking_models', {})
        self.model_name = []
        
        
        if logger is None:
            self.logger = self._get_default_logger()
        else:
            self.logger = logger

        self._download_models()
        self._load_models()
            
    def load_config(self, path):
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    
    def _get_default_logger(self):
        logger = logging.getLogger('EmbedRerankBuilder')
        logger.addHandler(logging.NullHandler())
        return logger
    
    def _download_models(self):
        for key, config in self.embedding_model_configs.items():
            model_directory = os.path.dirname(config['model_path'])
            tokenizer_directory = os.path.dirname(config['tokenizer_path'])
            os.makedirs(model_directory, exist_ok=True)
            os.makedirs(tokenizer_directory, exist_ok=True)
            
            self._download_model_or_tokenizer(key, config, 'model', AutoModel)
            self._download_model_or_tokenizer(key, config, 'tokenizer', AutoTokenizer)
        
        for key, config in self.reranking_model_configs.items():
            model_directory = os.path.dirname(config['model_path'])
            tokenizer_directory = os.path.dirname(config['tokenizer_path'])
            os.makedirs(model_directory, exist_ok=True)
            os.makedirs(tokenizer_directory, exist_ok=True)
            
            self._download_model_or_tokenizer(key, config, 'model', AutoModelForSequenceClassification)
            self._download_model_or_tokenizer(key, config, 'tokenizer', AutoTokenizer)
    
    def _download_model_or_tokenizer(self, key, config, type_, class_):
        path_key = f'{type_}_path'
        if not os.path.exists(config[path_key]):
            self.logger.info(f"[EmbedRerankBuilder] Downloading {type_} {key} to {config[path_key]}")
            try:
                obj = class_.from_pretrained(config['model_name'])
                obj.save_pretrained(config[path_key])
                del obj
            except Exception as e:
                self.logger.error(f"[EmbedRerankBuilder] Error downloading {type_} {key}: {e}")
        else:
            self.logger.info(f"[EmbedRerankBuilder] {type_.capitalize()} {key} already exists at {config[path_key]}")
    
    def _load_models(self):
        
        for key, config in self.embedding_model_configs.items():
            self.logger.info(f"[EmbedRerankBuilder] Loading model {key}")
            model = Embedder(
                model_name=key,
                model_path=config["model_path"],
                tokenizer_path=config["tokenizer_path"],
                use_gpu=config["use_gpu"],
                use_float16=config["use_float16"],
                max_length=config["max_length"],
                logger=self.logger
            )
            setattr(self, key, model)
            self.model_name.append(key)
        
        for key, config in self.reranking_model_configs.items():
            self.logger.info(f"[EmbedRerankBuilder] Loading model {key}")
            model = Reranker(
                model_name=key,
                model_path=config["model_path"],
                tokenizer_path=config["tokenizer_path"],
                use_gpu=config["use_gpu"],
                use_float16=config["use_float16"],
                max_length=config["max_length"],
                logger=self.logger
            )
            setattr(self, key, model)
            self.model_name.append(key)

    