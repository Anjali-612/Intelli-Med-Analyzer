import torch

cp = torch.load("medical_model.pth", map_location="cpu")

print("\n==== CHECKPOINT KEYS ====\n")
for k in cp.keys():
    print("-", k)

print("\n==== MODEL STATE KEYS (first 30) ====\n")
sd = cp.get("state_dict", cp)
for i,k in enumerate(sd.keys()):
    if i>30: break
    print(k)

print("\n==== CLASS NAMES IF PRESENT ====\n")
print(cp.get("class_names"))
