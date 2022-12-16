from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, get_scheduler
from datasets import load_dataset

import torch
import numpy as np
import random
from tqdm.auto import tqdm

# set up some seeds so that we can reproduce results
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Function to load the model and tokenizer
def load_model(bert):
  model = DistilBertForQuestionAnswering.from_pretrained(bert)
  tokenizer = DistilBertTokenizerFast.from_pretrained(bert)
  return model, tokenizer


# Function to load data
def load_data(path):
  dataset = load_dataset(path)
  return dataset['train'], dataset['validation']


# Custom dataset class to properly initialize Dataloader
class QADataset(torch.utils.data.Dataset):
  def __init__(self, dataset):
    self.dataset = dataset
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, index):
    instance = self.dataset[index]
    return {
        'input_ids': torch.tensor(instance['input_ids']),
        'attention_mask': torch.tensor(instance['attention_mask']),
        'ans_start': torch.tensor(instance['ans_start']),
        'ans_end': torch.tensor(instance['ans_end'])
    }


# Function that processes data and returns dataloader
def preprocess_and_tokenize(dataset, batch_size, tokenizer):

  # Processing function to map over the dataset
  def preprocess_function(example):
    # Set up text input
    question = f"[CLS] {example['questions'][0]['input_text']} [SEP]"
    context = example['contexts']
    text = f"{question} {context} [SEP]"
    
    inputs = tokenizer(
      text,
      max_length=512, 
      return_offsets_mapping=True, 
      add_special_tokens=False,
      padding='max_length', 
      truncation=True
    )

    # Initialize start and end tokens
    inputs['ans_start'] = 0
    inputs['ans_end'] = 0

    # Find new positions of start and end in the above text
    char_start = example['answers'][0]['span_start'] + len(question) + 1
    char_end = example['answers'][0]['span_end'] + len(question) + 1
    
    # Use offset mapping to find start and end token indices
    for i, (start, end) in enumerate(inputs.pop('offset_mapping')):
      if char_start == start:
        inputs['ans_start'] = i
      if char_end == end:
        inputs['ans_end'] = i
        break
    
    return inputs

  remove_cols = dataset.column_names
  processed = dataset.map(preprocess_function, remove_columns=remove_cols)
  return torch.utils.data.DataLoader(QADataset(processed), batch_size)

# Function that trains the model
def train_loop(model, train_data_loader, validation_data_loader, num_epochs, learning_rate, device):

  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=len(train_data_loader) * num_epochs
  )

  train_losses = []
  val_losses = []

  for epoch in range(num_epochs):
    model.train()
    train_losses.append(0)
    val_losses.append(0)
    progress_bar = tqdm(range(len(train_data_loader)))

    for batch in train_data_loader:
      output = model(
        batch['input_ids'].to(device), 
        batch['attention_mask'].to(device), 
        start_positions=batch['ans_start'].to(device),
        end_positions=batch['ans_end'].to(device)
      )
      loss = output.loss
      train_losses[-1] += float(loss)

      model.zero_grad()
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
    
      del output
      del loss
      progress_bar.update(1)

    train_losses[-1] /= len(train_data_loader)
    
    # Evaluate on validation data
    model.eval()
    with torch.no_grad():
      for batch in validation_data_loader:
        output = model(
          batch['input_ids'].to(device), 
          batch['attention_mask'].to(device), 
          start_positions=batch['ans_start'].to(device),
          end_positions=batch['ans_end'].to(device)
        )
        val_losses[-1] += float(output.loss)
    
    val_losses[-1] /= len(validation_data_loader)
    print(f"epoch {epoch + 1}")
    print(f"training loss: {train_losses[-1]}")
    print(f"validation loss: {val_losses[-1]}")

  return train_losses, val_losses
    

# Function that evaluates the model
def eval_loop(model, validation_data_loader, batch_size, device):

  model.eval()
  precisions = []
  recalls = []
  f1s = []
  
  with torch.no_grad():
    for batch in validation_data_loader:
      output = model(
        batch['input_ids'].to(device), 
        batch['attention_mask'].to(device)
      )

      for i in range(batch_size):
        if i >= len(batch):
          break
        
        # Predicted
        s_prime = output.start_logits[i].argmax().item()
        e_prime = output.end_logits[i].argmax().item()
        
        # Ground truth
        s = batch['ans_start'][i].item()
        e = batch['ans_end'][i].item()
 
        # Number of matching tokens
        if s_prime > e or e_prime < s:
          matched = 0
        else:
          matched = min(e, e_prime) - max(s, s_prime) + 1
        
        # Lengths of the intervals
        orig_len = e - s + 1
        bert_len = e_prime - s_prime + 1
  
        if bert_len > 0:
          precisions.append(matched / bert_len)
        else:
          precisions.append(0)
        
        recalls.append(matched / orig_len)
    
        if matched > 0 and bert_len > 0:
          f1s.append(2 / ((1 / precisions[-1]) + (1 / recalls[-1])))
        else:
          f1s.append(0)
  
  precision = sum(precisions) / len(precisions)
  recall = sum(recalls) / len(recalls)
  f1 = sum(f1s) / len(f1s)

  return precision, recall, f1


def main():
  '''Here's the basic structure of the main block -- feel free to add or
  remove parameters/helper functions as you see fit, but all steps here are 
  needed and we expect to see precision, recall, and f1 scores printed out'''
  device = "cuda" if torch.cuda.is_available() else "cpu"
  batch_size = 16
  num_epochs = 3
  learning_rate = 3e-5

  model, tokenizer = load_model("distilbert-base-uncased")
  train, validation = load_data("cjlovering/natural-questions-short")

  train_data_loader = preprocess_and_tokenize(train, batch_size, tokenizer)
  validation_data_loader = preprocess_and_tokenize(validation, batch_size, tokenizer)

  model.to(device)

  train_losses, val_losses = train_loop(model,
                                        train_data_loader, 
                                        validation_data_loader, 
                                        num_epochs,
                                        learning_rate,
                                        device)

  precision, recall, f1_score  = eval_loop(model, 
                                           validation_data_loader, 
                                           batch_size, 
                                           device)
  
  print("PRECISION: ", precision)
  print("RECALL: ", recall)
  print("F1-SCORE: ", f1_score)

if __name__ == "__main__":
  main()
