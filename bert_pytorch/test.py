import unittest
import torch
from model.bert import BERT


# bert_dense_meta_t x_meta(0, false, batch_size, seq_len, hidden_layer_size);
# bert_dense_meta_t mask_meta(0, false, batch_size, seq_len, seq_len);

bert = BERT()

batch_size = 1
seq_len = 20
hidden_layer_size = 768

for num_heads in range(12, 40):
    x = torch.rand(batch_size, seq_len, hidden_layer_size)
    mask = torch.rand(batch_size, seq_len, seq_len)

    for param_tensor in bert.state_dict():
        print(param_tensor, "\t", bert.state_dict()[param_tensor].size())

    torch.save(bert.state_dict(), "./bert.pth")
    bert.load_state_dict(torch.load("./bert.pth"))
    bert.eval()

    module = torch.jit.trace(bert, (x, mask)).cpu()
    torch.jit.save(module, 'bert-%s-heads-%s-hidden.pt' %
                   (num_heads, hidden_layer_size))
