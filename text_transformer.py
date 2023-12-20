import os
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Way better with gpt3 or above!

class SceneDescriptionGenerator:
    def __init__(self, model_name='gpt-3.5-turbo-instruct-0914', is_openai_api=True):
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

    def generate_description(self, objects_count, relationships, colors, similar_groups=None):
        # Construct input text with additional parameters
        input_text = f"There are {objects_count} objects in the scene. "
        num = 1
        prev_color = colors[0]
        for i, color in enumerate(colors[1:]):
            if color == prev_color:
                num += 1
            else:
                input_text += f"There are {num} adjancent {prev_color} items. "
                num = 1
            prev_color = color
        
        # Describe similarity groups
        if similar_groups:
            for group in similar_groups:
                input_text += f"There are {group['count']} similar {group['color']} items."
        
        # Add relationships
        input_text += " ".join([f"{rel[0]} is {rel[1]} of {rel[2]}" for rel in relationships])
        
        # Prompt for detailed paragraph
        input_text += " Describe the arrangement of these objects on the supermarket shelf in a detailed paragraph."

        if self.is_openai_api:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=input_text[:4097-150],
                max_tokens=150
            )
            generated_description = response.choices[0].text.strip()
        else:
            # Local generation using transformers
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            output = self.model.generate(input_ids, max_length=400, num_beams=5, early_stopping=True)
            generated_description = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_description
