import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


######################
# CONSTANTS TO ADJUST
model_name = "facebook/opt-125m"

######################

# load the model and tokenizers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
