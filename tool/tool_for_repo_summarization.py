from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import time
import argparse
import os


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test the specified summarization model with the provided training and test data.')
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('-s', '--string', default=None, help='String to summarize. For longer sequences use the flag -f | --file')
  group.add_argument('-f', '--file', default=None, help='Path of the file containing the text to summarize.')
  parser.add_argument('-m', '--model', default='./my_model', help='String of the model\'s name or path to its location.')
  parser.add_argument('-o', '--output', default=None, help='Path to where the summarized text will be stored.')
  parser.add_argument('-g', '--gpu', action=argparse.BooleanOptionalAction, help='Path to where the summarized text will be stored.')

  args = parser.parse_args()

  use_gpu = args.gpu
  output_dir = args.output
  input_string =  args.string
  if input_string is None:
    file_location = args.file
    with open(file_location, 'r') as f:
      input_string = f.read()

  results = {}

  model_name = args.model
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  if use_gpu:
    summarizer = pipeline("summarization", tokenizer=tokenizer, model=model, truncation=True, device="cuda:0")

  else:
    summarizer = pipeline("summarization", tokenizer=tokenizer, model=model, truncation=True)

  start_time = time.perf_counter()
  summary = summarizer(input_string)
  elapsed_time = time.perf_counter() - start_time
  results = {'time': elapsed_time, 'summary': summary[0]['summary_text']}

  print(f"Elapsed time: {results['time']} s")
  print(f"Summary:\n{results['summary']}")

  if output_dir is not None:
    with open(os.path.join(output_dir, 'output_summary.txt'), 'w') as f:
      f.write(results['summary'])