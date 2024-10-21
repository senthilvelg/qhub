import time
from symbol import parameters
from torchstat import statistics, compute_flops, compute_memory, stat
import requests
import torch
from starlette.requests import Request
from typing import Dict

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from ray import serve


# QHub Ray Serve Application


class QHubModelDeployment:
    def __init__(self, request_msg: str):
        self._model_name = "NECOUDBFM/Jellyfish"
        #self._model_name = "microsoft/phi-1_5"
        self._request_msg = request_msg
        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        print(f"Using device {self._device}")
        start_time = time.time()

        self._system_message = "You are an AI assistant that follows instruction extremely well."
        self._tokenizer =  AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype="auto",
            device_map=self._device,
            trust_remote_code=True
        )

        print(f"Model: {self._model}")
        num_parameters = sum(p.numel() for p in self._model.parameters())
        print(f"Num Parameters: {num_parameters}")
        print(f"Model init time: {(time.time() - start_time)} seconds")
        #stat(self._model, (1, 20))

    def call(self, request):
        prompt = f"{self._system_message}\n\n[INST]:\n\n{request}\n\n[\INST]"
        print(f"Prompt: {prompt}")
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)
        print(f"Input Ids Shape: {input_ids.shape}")

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
start_time = time.time()
model = QHubModelDeployment(request_msg = "Model Init")
event_list = ("[ { \"event_id\": \"12345\", \"timestamp\": \"2024-10-18T10:15:30Z\", \"user_id\": \"abc123\", \"action\": "
          "\"login\", \"metadata\": { \"location\": \"USA\", \"device\": \"mobile\" } },{ \"event_id\": \"12346\", "
          "\"timestamp\": \"2024-10-18T10:20:45Z\", \"user_id\": \"def456\", \"action\": \"purchase\", \"metadata\": "
          "{ \"location\": \"Canada\", \"device\": \"desktop\", \"amount\": 150.75, \"currency\": \"USD\" } },"
          "{ \"event_id\": \"12347\", \"timestamp\": \"2024-10-18T10:25:10Z\", \"user_id\": \"ghi789\", \"action\": "
          "\"logout\", \"metadata\": { \"location\": \"UK\", \"device\": \"tablet\" } } ]")
query = f"Here is  a list of events. Are there any schema changes in the events? events list: {event_list}"
app = model.call(request=query)
print(f"Finished: {app} in {(time.time() - start_time)} seconds")

