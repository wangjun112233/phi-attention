"""Quick diagnostic: check Qwen2Config RoPE attribute names"""
import transformers
print("transformers version:", transformers.__version__)
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
attrs = [a for a in dir(config) if not a.startswith("_") and ("rope" in a.lower() or "theta" in a.lower() or "rotary" in a.lower())]
print("Rope-related attrs:", attrs)
d = config.to_dict()
for k,v in d.items():
    if "rope" in k.lower() or "theta" in k.lower() or "rotary" in k.lower():
        print(f"  {k} = {v}")
