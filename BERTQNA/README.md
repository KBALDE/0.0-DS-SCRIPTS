# Question Answering using BERT
To use this script you need to install some libraries
```bash
pip install datasets evaluate transformers[sentencepiece]
pip install accelerate
```

And then you need to prepare your dataset. And that you may choose to work sample first. Or at least prepare your dataset and store it to disk.

```python
import datasets

def sample_dataset(ds, size=100):
    for i in ds.keys():
        ds[i]=ds[i].shuffle(seed=42).select(range(size))
    return ds

def save_dataset(ds_sample_name):
    ds.save_to_disk(ds_sample_name)
    print("The Dataset is saved")
```    
  
### Training 
```python
python ./BertQuestionAnswer.py --input_dataset="squad-sample" --max_length=384  --stride=128 --n_best=20  --max_answer_length=30 \
                                --model_checkpoint="bert-base-uncased" \
                                --metric_data_load="squad" --output_dir="bert-question-answering" --num_train_epochs=5 --learning_rate=2e-5 
```

### Inference
```python
python InferenceForBertQuestionAnswering.py --model_checkpoint="bert-question-answering" \
                                            --question="Which deep learning libraries back Transformers?"\
                                            --context="""Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other."""
```

