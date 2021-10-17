import unittest
import torch
from model.bert import BERT


# bert_dense_meta_t x_meta(0, false, batch_size, seq_len, hidden_layer_size);
# bert_dense_meta_t mask_meta(0, false, batch_size, seq_len, seq_len);

bert = BERT()

batch_size = 1
seq_len = 20
num_heads = 12
hidden_layer_size = 768

x = torch.rand(batch_size, seq_len, hidden_layer_size)
mask = torch.rand(batch_size, seq_len, seq_len)


for param_tensor in bert.state_dict():
    print(param_tensor, "\t", bert.state_dict()[param_tensor].size())

torch.save(bert.state_dict(), "./bert.pth")
bert.load_state_dict(torch.load("./bert.pth"))
bert.eval()


module = torch.jit.trace(bert, (x, mask)).cpu()
torch.jit.save(module, 'scriptmodule.pt')