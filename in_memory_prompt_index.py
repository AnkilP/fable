import faiss
import numpy as np
import torch
from dataclasses import dataclass
from typing import Union, List


@dataclass
class MetadataInput:
    prompt: Union[str, List[str]]
    vector_field: Union[torch.Tensor, np.ndarray]


@dataclass
class MetadataOutput:
    prompt: Union[str, List[str]]
    associated_prompt: Union[str, List[str]]
    vector_field: Union[torch.Tensor, np.ndarray, dict]
    distance: float


class InMemoryPromptIndex:
    def __init__(self, embedding_dim=512, hnsw_m=32):
        self.embedding_dim = embedding_dim
        self.hnsw_m = hnsw_m
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m)
        self.metadata_store = {}

    def add_prompt(self, prompt, obj):
        faiss_prompt = self._to_faiss_compatible(prompt)
        self.index.add(np.expand_dims(faiss_prompt, axis=0))
        vector_id = self.index.ntotal - 1
        self.metadata_store[vector_id] = MetadataInput(prompt=prompt, vector_field=obj)

    def get_prompt(self, prompt):
        faiss_query = self._to_faiss_compatible(prompt)
        distances, indexes = self.index.search(np.expand_dims(faiss_query, axis=0), k=1)
        # todo: k = 3 can allow us to do HMM approximation and eliminate compound error in the vector
        results = []
        for distance, idx in zip(distances[0], indexes[0]):
            if idx != -1:
                metadata = self.metadata_store.get(idx, {})
                results.append(MetadataOutput(prompt=prompt, associated_prompt=metadata.prompt, vector_field=metadata.vector_field, distance=distance))
        return results

    @staticmethod
    def _to_faiss_compatible(embedding):
        """
        Converts PyTorch Tensor or FloatTensor to NumPy float32 for FAISS.
        """
        if isinstance(embedding, torch.Tensor):
            return embedding.detach().cpu().numpy().astype('float32')
        elif isinstance(embedding, np.ndarray):
            return embedding.astype('float32')
        else:
            raise TypeError("Embedding must be a PyTorch Tensor or NumPy array.")
