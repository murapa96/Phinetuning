import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

# Cargar el modelo y el tokenizador
model = AutoModelForCausalLM.from_pretrained("ruta/a/tu/modelo")
tokenizer = AutoTokenizer.from_pretrained("ruta/a/tu/modelo")

# Asegurarse de que el modelo esté en modo de evaluación
model.eval()

# Si tienes una GPU disponible, mover el modelo a la GPU
if torch.cuda.is_available():
    model = model.to("cuda")

# Generar un texto de entrada
input_text = "El inicio de tu texto"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Si tienes una GPU disponible, mover los IDs de entrada a la GPU
if torch.cuda.is_available():
    input_ids = input_ids.to("cuda")

# Generar texto
output = model.generate(input_ids, max_length=100, temperature=0.7)

# Decodificar el texto generado
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)