import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import os

class CULBot:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", adapter_path="./adapter"):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuración para CPU por defecto para Railway (si no hay GPU)
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        
        # Cargamos el modelo base
        # Nota: En Railway sin GPU no usamos BitsAndBytesConfig ya que requiere CUDA
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=device_map,
            )

        # Cargamos el adaptador LoRA si existe
        if os.path.exists(adapter_path):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print("✅ Adaptador LoRA cargado correctamente.")
            
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.system_prompt = "Eres el Asistente Virtual de Gestión Académica de la Corporación Universitaria Latinoamericana (CUL). Responde siempre en español, de forma formal, amable y concisa. Solo ayudas con trámites académicos."

    def generate_response(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        generated_text = outputs[0]['generated_text']
        response = generated_text.split("<|assistant|>")[-1].strip()
        return response
