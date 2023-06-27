from falcontune.model import load_model
from falcontune.model.lora import load_adapter
from transformers import StoppingCriteriaList, StoppingCriteria
import torch

class AMPWrapper:
    def __init__(self, model, options=None):
        self.model = model
        self.options = options
        if self.options is None:
            self.options = {'enabled': True, 'device_type': 'cuda'}

    def autocast_forward(self, *args, **kwargs):
        with torch.amp.autocast(**self.options):
            return self.model.non_autocast_forward(*args, **kwargs)

    def autocast_generate(self, *args, **kwargs):
        with torch.amp.autocast(**self.options):
            return self.model.non_autocast_generate(*args, **kwargs)

    def apply_forward(self):
        self.model.non_autocast_forward = self.model.forward
        self.model.forward = self.autocast_forward

    def apply_generate(self):
        self.model.non_autocast_generate = self.model.generate
        self.model.generate = self.autocast_generate

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if stop == input_ids[0][-1]:
                return True

        return False

def init_falcon(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    lora_type: str = "lora"
):
  print(f"Loading falcon {base_model} with {lora_weights}")

  model, tokenizer = load_model(
        'falcon-7b',
        base_model
  )

  model = load_adapter(model, lora_apply_dir=lora_weights)

  wrapper = AMPWrapper(model)
  wrapper.apply_generate()

  stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=[tokenizer.eos_token_id])])

  return (model, tokenizer, stopping_criteria)
