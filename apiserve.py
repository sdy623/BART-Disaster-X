from fastapi import FastAPI
import torch
from transformers import pipeline

from ray import serve
from ray.serve.handle import DeploymentHandle
from inference import DisasterClassifier, CustomTextClassificationPipeline
from starlette.requests import Request
app = FastAPI()
MODEL_PATH = "./model"


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
)
class DisasterClassifier:
    def __init__(self):
        self.classifier = CustomTextClassificationPipeline(model_path=MODEL_PATH)

    def classify(self, sentence: str):
        return self.classifier.predict(sentence)
        
    async def __call__(self, http_request: Request) -> str:
        try:
            text: str = await http_request.json()
        except ValueError:
            return {"error": "Invalid JSON"}
        return self.classify(text)


entrypoint = DisasterClassifier.bind()
