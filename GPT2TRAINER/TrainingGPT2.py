import evaluate
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import datasets

import math

from transformers import Trainer, TrainingArguments

import argparse

parser=argparse.ArgumentParser()

parser.add_argument("--training_filename", type=str, help="d_name_tuple")
parser.add_argument("--input_name", type=str, help="input_name")
parser.add_argument("--train_split_name", type=str, help="train_split_name")
parser.add_argument("--valid_split_name", type=str, help="valid_split_name")
parser.add_argument("--test_split_name", type=str, help="test_split_name")
parser.add_argument("--model_checkpoint", type=str, help="model_checkpoint")

parser.add_argument("--context_length", type=int, help="context_length")
parser.add_argument("--block_size", type=int, help="block_size")
parser.add_argument("--num_epochs", type=int, help="num_epochs")
parser.add_argument("--learning_rate", type=float, help="learning_rate")

parser.add_argument("--weight_decay", type=float, help="weight_decay")
parser.add_argument("--evaluation_strategy", type=str, help="evaluation_strategy")
parser.add_argument("--bool_hub", type=bool, help="bool_hub")
parser.add_argument("--training_dir", type=str, help="training_dir")
parser.add_argument("--inference_dir", type=str, help="inference_dir")
args=parser.parse_args()

training_filename=args.training_filename
input_name=args.input_name
train_split_name=args.train_split_name
valid_split_name=args.valid_split_name
test_split_name=args.test_split_name
model_checkpoint=args.model_checkpoint
context_length=args.context_length
block_size=args.block_size
num_epochs=args.num_epochs
learning_rate=args.learning_rate
weight_decay=args.weight_decay
evaluation_strategy=args.evaluation_strategy
training_dir=args.training_dir
inference_dir=args.inference_dir
bool_hub=args.bool_hub



def main():
    
    # data processors
    def function_tokenize(element):
        outputs = tokenizer(
            element[input_name],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )

        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}




    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # load model and tokenizer
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print("Tokenizer and Model are loaded")

    training_args = TrainingArguments(
        output_dir=training_dir,
        overwrite_output_dir=True,
        evaluation_strategy= evaluation_strategy,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        #push_to_hub=bool_hub
    )
    
    # read and sample
    raw_ds= datasets.load_from_disk(training_filename)
    
    train_val_set = raw_ds.map(
        function_tokenize, 
        batched=True, 
        remove_columns=raw_ds[train_split_name].column_names
    )
    
    train_val_set = train_val_set.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_val_set[train_split_name],
        eval_dataset=train_val_set[valid_split_name],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # train the model
    trainer.train()

    trainer.save_model(inference_dir)



if __name__=="__main__":
    main()
