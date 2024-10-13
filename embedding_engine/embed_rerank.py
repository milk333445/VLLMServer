import torch

from transformers import AutoModelForSequenceClassification, AutoModel

from embedding_engine.baseinferencer import BaseInferencer


class Reranker(BaseInferencer):
    def load_model(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)
    
    def rerank(self, query, documents, batch_size=8): 
        all_scores = []
        all_documents = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            pairs = [[query, doc] for doc in batch_docs]
            
            with torch.no_grad():
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
                
                if self.device.type == 'cuda':
                    scores = scores.cpu()
                scores = scores.numpy()
                
                all_scores.extend(scores)
                all_documents.extend(batch_docs)
                
                del inputs, scores
                torch.cuda.empty_cache()
        return all_scores
    
    
class Embedder(BaseInferencer):
    def load_model(self, model_path):
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)
        
    def get_embeddings(self, texts, batch_size=8):
        all_embeddings = []
        if not isinstance(texts, list):
            texts = [texts]
        for  i in range(0, len(texts), batch_size):
            self.logger.debug(f"[Embedder] model: {self.model_name} embedding {len(texts)} texts")
            batch_texts = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu())
            
            del outputs, embeddings, encoded_input
            torch.cuda.empty_cache()
            
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        return final_embeddings.numpy()
