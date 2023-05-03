# Data preparation
I use utils script to load dataset and sample it

# Training
```python 
!python TrainingGPT2.py --training_filename="tiny-wiki" \
                     --input_name="text" \
                     --train_split_name="train" \
                     --valid_split_name="validation" \
                     --test_split_name="test" \
                     --model_checkpoint="gpt2" \
                     --context_length=128 \
                     --block_size=128 \
                     --num_epochs=10 \
                     --learning_rate=2e-4 \
                     --weight_decay=0.01 \
                     --evaluation_strategy="epoch" \
                     --training_dir="gpt2-training" \
                     --inference_dir="gpt2-inference" 
```

# Inference

```python
from transformers import pipeline
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pipe = pipeline(
    "text-generation", model="gp2-inference",
)
txt="Life is"

gen = pipe(txt, num_return_sequences=1)

print(gen[0]['generated_text'])
```
