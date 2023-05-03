
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--input_dataset", type=str, help="input_dataset")
parser.add_argument("--max_length", type=int, help="max_length")
parser.add_argument("--stride", type=int, help="stride")
parser.add_argument("--n_best", type=int, help="n_best")
parser.add_argument("--max_answer_length", type=int, help="max_answer_length")
parser.add_argument("--model_checkpoint", type=str, help="model_checkpoint")
parser.add_argument("--metric_data_load", type=str, help="metric_data_load")
parser.add_argument("--output_dir", type=str, help="output_dir")
parser.add_argument("--num_train_epochs", type=int, help="num_train_epochs")
parser.add_argument("--learning_rate", type=float, help="learning_rate")

args = parser.parse_args()

input_dataset=args.input_dataset
max_length=args.max_length
stride=args.stride
n_best=args.n_best
max_answer_length=args.max_answer_length
model_checkpoint=args.model_checkpoint
metric_data_load=args.metric_data_load
output_dir=args.output_dir
num_train_epochs=args.num_train_epochs
learning_rate=args.learning_rate


from tqdm.auto import tqdm
import collections
import numpy as np

import argparse 
from transformers import AutoTokenizer
import evaluate
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering

from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
import numpy as np
import datasets



# UTILS FUNCTIONS

def tokenize_inputs(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    return inputs

def preprocess_training(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs



def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)



def train_eval_data_loader(train_set, eval_set, batch_size = 64):
    
    train_dl = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )
    eval_dl = DataLoader(
        eval_set, 
        batch_size=batch_size, 
        collate_fn=default_data_collator
    )
    
    return train_dl, eval_dl


# MODELS

def main():

    # load model, tokenizer and metric 
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    
    metric = evaluate.load(metric_data_load)

    ### TK 
    def preprocess_training(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
  
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
  
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
  
          # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
                

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs


    def preprocess_validation(examples):
      questions = [q.strip() for q in examples["question"]]
      inputs = tokenizer(
          questions,
          examples["context"],
          max_length=max_length,
          truncation="only_second",
          stride=stride,
          return_overflowing_tokens=True,
          return_offsets_mapping=True,
          padding="max_length",
      )

      sample_map = inputs.pop("overflow_to_sample_mapping")
      example_ids = []

      for i in range(len(inputs["input_ids"])):
          sample_idx = sample_map[i]
          example_ids.append(examples["id"][sample_idx])

          sequence_ids = inputs.sequence_ids(i)
          offset = inputs["offset_mapping"][i]
          inputs["offset_mapping"][i] = [
              o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
          ]

      inputs["example_id"] = example_ids
      return inputs




    
    print("model, tokenizer,a nd metric loading is done!", tokenizer)
    
    # read raw_dataset
    raw_datasets= datasets.load_from_disk(input_dataset)

    # preprocess data
    train_dataset = raw_datasets["train"].map(
            preprocess_training,
            batched=True,
            remove_columns=raw_datasets["train"].column_names
    )

    val_dataset = raw_datasets["validation"].map(
                preprocess_validation,
                batched=True,
                remove_columns=raw_datasets["validation"].column_names,
    )
    
    val_dataset = val_dataset.remove_columns(["example_id", "offset_mapping"])


    # train_dataset, val_dataset = tokenize_process(raw_datasets,
    #                                               preprocess_training, 
    #                                               preprocess_validation)
    print("train data is preprocessed", train_dataset)
    print("val data is preprocessed", val_dataset)
    
    # dataloader wrapper
    train_dataloader, eval_dataloader = train_eval_data_loader(train_dataset, 
                                                               val_dataset, 
                                                               batch_size = 8)
    
  
    print("dataloader wrapped")
  
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
  
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
  
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
  
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
  
    print("Start training!")
    progress_bar = tqdm(range(num_training_steps))
  
    # TRAINING
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
  
        # Evaluation
        model.eval()
        start_logits = []
        end_logits = []
        accelerator.print("Evaluation!")
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
  
            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
  
        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(val_dataset)]
        end_logits = end_logits[: len(val_dataset)]

        # start_logits, end_logits, features, examples
  
        #metrics = compute_metrics( # use of raw ds. and val_ds
        #    start_logits, end_logits, val_dataset, raw_datasets["validation"]
        #)
        #print(f"epoch {epoch}:", metrics)
  
        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

if __name__=="__main__":
    main()
  
  
