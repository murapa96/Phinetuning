import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

# Definir los argumentos de la línea de comandos
parser = argparse.ArgumentParser(description='Generate script')
parser.add_argument('--model_path', type=str, default="ruta/a/tu/modelo", help='Path to the model')
parser.add_argument('--tokenizer_path', type=str, default="ruta/a/tu/modelo", help='Path to the tokenizer')
parser.add_argument('--input_text', type=str, default="El inicio de tu texto", help='Input text to generate from')

# Parse the command line arguments
args = parser.parse_args()

# Cargar el modelo y el tokenizador
model = AutoModelForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

# Asegurarse de que el modelo esté en modo de evaluación
model.eval()

# Si tienes una GPU disponible, mover el modelo a la GPU
if torch.cuda.is_available():
    model = model.to("cuda")

# Generar un texto de entrada
input_ids = tokenizer.encode(args.input_text, return_tensors="pt")

# Si tienes una GPU disponible, mover los IDs de entrada a la GPU
if torch.cuda.is_available():
    input_ids = input_ids.to("cuda")

# Generar texto
output = model.generate(input_ids, max_length=100, temperature=0.7)

# Decodificar el texto generado
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)