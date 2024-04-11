import os
import openai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np


MAX_PROMPT_LEN = 16385
# Way better with gpt3 or above!

class SceneDescriptionGenerator:
    def __init__(self, model_name='gpt3.5-turbo', is_openai_api=True):
        # Determine if using OpenAI's API or a local transformers model
        self.model_name = model_name
        self.is_openai_api = is_openai_api
        if is_openai_api:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_description(self, scene_from_template, final_array):
        
        client = OpenAI()
        # Prompt for detailed paragraph
        prompt = "Summarize the following spatial relationships into a concise supermarket scene description: \n"
        
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert arrays to lists
            elif isinstance(obj, np.float32):
                return float(obj)  # Convert np.float32 to Python float
            elif isinstance(obj, np.int32):
                return int(obj)  # Convert np.float32 to Python float
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        print(len(final_array))
        # serialized_array = make_serializable(final_array)

        if self.is_openai_api:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": f"{prompt}\n{final_array}"}
                    ]
            )
            generated_description = response.choices[0].message.content
        else:
            # Local generation using transformers
            input_text = f"{prompt}\n\n{scene_from_template}\n{final_array}"
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            output = self.model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)
            generated_description = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_description
