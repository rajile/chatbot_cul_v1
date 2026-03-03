import torch
import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login


# 1. Configuración Inicial
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "/content/dataset_cul_mejorado_v2.jsonl"
DRIVE_DATASET_PATH = "/content/drive/MyDrive/CUL_AI_Model/dataset_cul_mejorado_v2.jsonl"
HF_TOKEN = "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"

# Fallback si no está en /content
if not os.path.exists(DATASET_PATH) and os.path.exists(DRIVE_DATASET_PATH):
    import shutil
    print("📂 Dataset no encontrado en /content, copiando desde Drive...")
    shutil.copy(DRIVE_DATASET_PATH, DATASET_PATH)

# Verificación de versión de bitsandbytes
try:
    import bitsandbytes as bnb
    from packaging import version
    if version.parse(bnb.__version__) < version.parse("0.46.1"):
        print(f"⚠️ ¡ATENCIÓN! Tu versión de bitsandbytes es {bnb.__version__}. Se requiere >= 0.46.1.")
        print("Ejecuta: !pip install -U bitsandbytes>=0.46.1")
        print("Luego ve a: Entorno de ejecución > Reiniciar sesión")
except ImportError:
    pass

login(token=HF_TOKEN)


# 2. Cargar Tokenizador y Configurar Chat Template
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. Preparar Dataset con el Chat Template de TinyLlama
# TinyLlama usa: <|system|>\n{system}<|endoftext|>\n<|user|>\n{user}<|endoftext|>\n<|assistant|>\n{assistant}
def format_with_chat_template(sample):
    messages = [
        {"role": "system", "content": sample['instruction']},
        {"role": "user", "content": sample.get('input', "Hola")},
        {"role": "assistant", "content": sample['output']}
    ]
    # Usamos tokenize=False para que devuelva el string formateado
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

data = []
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: data.append(json.loads(line))
else:
    print(f"❌ Error: No se encontró el dataset en {DATASET_PATH}")

dataset = Dataset.from_list(data)
dataset = dataset.map(format_with_chat_template)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset['train'].remove_columns([c for c in dataset['train'].column_names if c != 'text'])

print(f"✅ Dataset preparado: {len(train_ds)} ejemplos de entrenamiento.")

# 4. Cargar Modelo en 4-bit (Optimizado para T4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # T4 prefiere float16 sobre bfloat16
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# 5. Configurar LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 6. Preparar el Dataset para el Trainer Estándar (Más robusto)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

# Tokenizamos y configuramos etiquetas (labels = input_ids para CausalLM)
tokenized_ds = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_ds.set_format("torch")

# 7. Argumentos de Entrenamiento (Standard TrainingArguments)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="/content/cul_tinyllama_fixed",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    learning_rate=5e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    gradient_checkpointing=True,
    remove_unused_columns=False, # Importante para LoRA
)

# 8. Entrenar con el Trainer estándar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)




print("🚀 Iniciando entrenamiento corregido...")
trainer.train()

# 8. Guardar el Adaptador
model.save_pretrained("/content/cul_tinyllama_adapter")
print("✅ Entrenamiento completado y adaptador guardado.")
