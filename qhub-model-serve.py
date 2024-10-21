import requests
import torch
from starlette.requests import Request
from typing import Dict

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from ray import serve

# QHub Ray Serve Application
@serve.deployment
class QHubModelDeployment:
    def __init__(self, request_msg: str):
        self._model_name = "NECOUDBFM/Jellyfish"
        self._request_msg = request_msg
        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        print(f"Using device {self._device}")

        self._system_message = "You are an AI assistant that follows instruction extremely well."
        self._tokenizer =  AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def __call__(self, request: Request) -> Dict:
        prompt = f"{self._system_message}\n\n[INST]:\n\n{self._request_msg}\n\n[\INST]"
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.35,
            top_p=0.9
        )
        #return ",".join({f"{k},{v}" for k,v in request.items()})
        with torch.no_grad():
            generation_output = self._model.generate(
                input_ids=input_ids,
                generation_config = generation_config,
                return_dict_in_generate = True,
                output_scores=True,
                max_new_tokens=1024,
                pad_token_id=self._tokenizer.eos_token_id,
                repetition_penalty=1.15
            )

        output = generation_output[0]
        response = self._tokenizer.decode(
            output[:, input_ids.shape[-1]:][0], skip_special_tokens=True
        ).strip()

        return {"result": response}

app = QHubModelDeployment.bind(request_msg = "Welcome to QHub")

# Deploy Locally
serve.run(app, blocking=True, route_prefix="/")