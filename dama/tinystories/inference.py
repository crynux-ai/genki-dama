import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps"

model = AutoModelForCausalLM.from_pretrained("./checkpoints").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

model.eval()

input_ids = tokenizer.encode(sys.argv[1], return_tensors="pt").to(DEVICE)
output = model.generate(input_ids, max_length=500, do_sample=True)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
