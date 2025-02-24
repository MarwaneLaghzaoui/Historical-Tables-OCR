import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Charger le tokenizer et le modèle TableLlama
tokenizer = AutoTokenizer.from_pretrained("osunlp/TableLlama")
model = AutoModelForCausalLM.from_pretrained("osunlp/TableLlama")

# Exemple d'entrée sous forme de table (à adapter selon ton besoin)
table_text = """| Name  | Age | City      |
|-------|-----|----------|
| Alice |  25 | New York |
| Bob   |  30 | London   |
| Charlie | 27 | Paris  |"""

# Ajouter une instruction pour corriger les incohérences
input_text = f"Corrige les erreurs dans cette table :\n{table_text}"

# Tokenization
inputs = tokenizer(input_text, return_tensors="pt")

# Génération de texte
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=500)

# Décodage et affichage du résultat
corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(corrected_text)

