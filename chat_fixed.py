import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# 1. Configuración de Carga
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "/content/cul_tinyllama_adapter" # Ruta donde se guardó el adaptador

# 2. Cargar Tokenizador
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 3. Cargar Modelo con el Adaptador
# Cargamos en 4-bit para ahorrar VRAM durante la inferencia
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Cargar los pesos entrenados (LoRA)
if torch.cuda.is_available() and ADAPTER_PATH:
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print("✅ Adaptador LoRA cargado correctamente.")
    except Exception as e:
        print(f"⚠️ No se pudo cargar el adaptador: {e}. Usando modelo base.")

# 4. Pipeline de Generación
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def chat_cul():
    system_prompt = "Eres el Asistente Virtual de Gestión Académica de la Corporación Universitaria Latinoamericana (CUL). Responde siempre en español, de forma formal, amable y concisa. Solo ayudas con trámites académicos."
    
    print("\n💬 Chatbot CUL listo (Escribe 'salir' para terminar)")
    print("-" * 50)

    while True:
        user_input = input("👤 Estudiante: ")
        if user_input.lower() in ["salir", "exit", "chao"]:
            break

        # Formatear el prompt usando el template oficial
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        # add_generation_prompt=True añade el tag <|assistant|> al final
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Extraer solo la respuesta del asistente
        generated_text = outputs[0]['generated_text']
        response = generated_text.split("<|assistant|>")[-1].strip()
        
        print(f"🤖 CUL Bot: {response}")
        print("-" * 50)

if __name__ == "__main__":
    chat_cul()
