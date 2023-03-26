import argparse
from transformers import pipeline


parser = argparse.ArgumentParser()

parser.add_argument("--model_checkpoint", type=str, help="model_checkpoint")
parser.add_argument("--question", type=str, help="question")
parser.add_argument("--context", type=str, help="context")

args = parser.parse_args()

model_checkpoint=args.model_checkpoint
question=args.question
context=args.context 



def main():

  question_answerer = pipeline("question-answering", model=model_checkpoint)

  resp=question_answerer(question=question, context=context)['answer']

  print(resp)


if __name__=="__main__":
  main()
