import sys
from python_environment_check import check_packages
import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
from collections import Counter, OrderedDict
from torch.utils.data import DataLoader
# from torchdata.dataloader2 import DataLoader2
import time
from torchinfo import summary
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from transformers import GPT2Model
from transformers import GPT2Config
import torchtext
from torchtext import __version__ as torchtext_version
from pkg_resources import parse_version
from torch.profiler import profile, record_function, ProfilerActivity

# ## Package version checks

# Check recommended package versions

d = {
    'torch': '1.8.0',
    'torchtext': '0.10.0'
}
check_packages(d)


# Step 1: load and create the datasets

train_dataset = IMDB(split='train')
test_dataset = IMDB(split='test')
# https://pytorch.org/text/stable/datasets.html#imdb

torch.manual_seed(1)

# train_dataset, valid_dataset = random_split(list(train_dataset), [20000, 5000])

# DEBUG
train_dataset, valid_dataset = random_split(list(train_dataset)[0:500], [400, 100])

train_size = 20000
valid_size = 5000
test_size = 25000

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
config = GPT2Config.from_pretrained('gpt2')
# Account for added pad token
config.vocab_size = 50258
gpt_model = GPT2Model(config)
# gpt_model = GPT2Model.from_pretrained('gpt2')
print(gpt_model)
max_seq_len = gpt_model.config.n_positions

encoded_input = tokenizer(train_dataset[0][-1], return_tensors='pt')
out = gpt_model(**encoded_input)
hidden_size = out['last_hidden_state'].shape[2]
summary(gpt_model, input_data=[encoded_input["input_ids"]], device="cpu")

def encode_transform_batch(device, tokenizer, hidden_size):

    text_pipeline = lambda x: [sentence[1] for sentence in x]
    if parse_version(torchtext.__version__) > parse_version("0.10"):
        label_pipeline = lambda x: 1. if x == 2 else 0.  # 1 ~ negative, 2 ~ positive review
    else:
        label_pipeline = lambda x: 1. if x == 'pos' else 0.

    def collate_fn(batch):
        encoded_inputs = text_pipeline(batch)
        encoded_inputs = tokenizer(encoded_inputs, return_tensors='pt', max_length = min(max_seq_len-1, len(max(encoded_inputs))), padding=True, truncation=True)
        # encoded_inputs = tokenizer(encoded_inputs, return_tensors='pt', truncation=True)
        # encoded_inputs.to(device)
        # model.to("cpu")
        # encoded_inputs.to("cpu")
        # out = model(**encoded_inputs)
        # all_data = out['last_hidden_state'][:,-1,:]
        label_list = []
        # all_data_tensor = torch.empty((0, hidden_size), dtype=torch.float32)
        # all_data_tensor = torch.cat((all_data_tensor.to(device), out['last_hidden_state'][:,-1,:].detach()), 0)
        # all_data_tensor = torch.empty((0, max_seq_len-1, hidden_size), dtype=torch.float32)
        # all_data_tensor = torch.cat((all_data_tensor.to(device), out['last_hidden_state'].detach()), 0)
        # all_data_tensor.to(device)
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            # encoded_input = tokenizer(_text, return_tensors='pt')
            # # processed_text = model(**encoded_input)
            # encoded_input.to(device)
            # processed_text = model(input_ids = encoded_input["input_ids"][:, :max_seq_len-1], attention_mask = encoded_input["attention_mask"][:, :max_seq_len-1])
            # # output = torch.tensor(processed_text['last_hidden_state'][:,-1,:])
            # output = processed_text['last_hidden_state'][:,-1,:].detach()
            # all_data_tensor = torch.cat((all_data_tensor.to(device), output), 0)
            # text_list.append(output.squeeze(0))
        label_list = torch.tensor(label_list)
        # padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        # return padded_text_list.to(device), label_list.to(device)
        return encoded_inputs.to(device), label_list.to(device)
    
    return collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_model.to(device)
collate_fn = encode_transform_batch(device, tokenizer, hidden_size)



## Step 4: batching the datasets

batch_size = 16

train_dl = DataLoader(train_dataset, batch_size=batch_size,
                      shuffle=True, collate_fn=collate_fn)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size,
                      shuffle=False, collate_fn=collate_fn)
test_dl = DataLoader(test_dataset, batch_size=batch_size,
                     shuffle=False, collate_fn=collate_fn)



class MLP(nn.Module):
    def __init__(self, input_dim, fc_hidden_size, loss_fn, gpt_model):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = loss_fn
        self.gpt_model = gpt_model
        self.gpt_model.eval()
        # self.optimizer = optimizer

    def forward(self, input):
        with torch.no_grad():
            x = self.gpt_model(**input)
        x = x['last_hidden_state'][:,-1,:]
        # x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x
    
    def train_procedure(self, dataloader):
        self.train()
        total_acc, total_loss = 0, 0
        for text_batch, label_batch in dataloader:
            self.optimizer.zero_grad()
            pred = self(text_batch)
            loss = self.loss_fn(pred.squeeze(-1), label_batch)
            loss.backward()
            self.optimizer.step()
            total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)
        return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)
    
    def evaluate_procedure(self, dataloader, size):
        self.eval()
        total_acc, total_loss = 0, 0
        with torch.no_grad():
            for text_batch, label_batch in dataloader:
                pred = self(text_batch)
                loss = self.loss_fn(pred.squeeze(-1), label_batch)
                total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
                total_loss += loss.item()*label_batch.size(0)
        # return total_acc/size, total_loss/size
        return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)
    
    def train_epochs(self, num_epochs, train_dl, valid_dl, valid_size):
        for epoch in range(num_epochs):
            start_time = time.time()
            acc_train, loss_train = self.train_procedure(train_dl)
            acc_valid, loss_valid = self.evaluate_procedure(valid_dl, valid_size)
            end_time = time.time()
            print(f'Epoch {epoch} accuracy: {acc_train:.4f} val_accuracy: {acc_valid:.4f} time: {end_time-start_time}')


loss_fn = nn.BCELoss()
model = MLP(hidden_size, 1024, loss_fn, gpt_model)    
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
model.optimizer = optimizer


print(model) 
 
model.to(device)
num_epochs = 5
model.train_epochs(num_epochs, train_dl, valid_dl, valid_size)

# DEBUG
# with profile(activities=[
#         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("model_training"):
#         model.train_epochs(num_epochs, train_dl, valid_dl, valid_size)

# print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))