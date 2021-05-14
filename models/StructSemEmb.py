import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StructSemEmbEncoder(nn.Module):

    def __init__(self, emb_dim, hidden_dim, item_num, user_num, vocab_size, rating_range):
        super(StructSemEmbEncoder, self).__init__()
        self.is_cuda_available = False
        if torch.cuda.is_available():
            print('CUDA/GPU device is available: [{}]'.format(torch.cuda.get_device_name(0)))
            self.is_cuda_available = True

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.item_num = item_num
        self.user_num = user_num
        self.vocob_size = vocab_size
        self.rating_range = rating_range

        self.item_embeddings = nn.Embedding(item_num, emb_dim)
        if self.is_cuda_available:
            self.item_embeddings = self.item_embeddings.cuda()

        self.user_embeddings = nn.Embedding(user_num, emb_dim)
        if self.is_cuda_available:
            self.user_embeddings = self.user_embeddings.cuda()

        self.word_embeddings = nn.Embedding(vocab_size, emb_dim)
        if self.is_cuda_available:
            self.word_embeddings = self.word_embeddings.cuda()

        self.gru_encoder = nn.GRU(emb_dim, hidden_dim)

        self.dropout = nn.Dropout(0.5)

        self.normalize_layer = nn.Linear(self.emb_dim * 2, self.emb_dim)

    def forward(self, item_id, item_desc):

        item_desc_word_embs = self.word_embeddings(torch.LongTensor(item_desc)).view(len(item_desc), 1, -1)

        gru_out, hidden_state = self.gru_encoder(item_desc_word_embs)
        hidden_state = hidden_state.view(self.emb_dim, 1, -1)
        item_emb = self.item_embeddings(torch.LongTensor([item_id])).view(self.emb_dim, 1, -1)

        combined_item_desc_emb = torch.cat([hidden_state, item_emb], 0).view(1, -1, self.emb_dim * 2)
        normalized_combined_item_desc_emb = self.normalize_layer(combined_item_desc_emb)
        return normalized_combined_item_desc_emb.view(self.emb_dim, 1, -1)


class StructSemEmbDecoder(nn.Module):

    def __init__(self, emb_dim, hidden_dim, item_num, user_num, vocab_size, rating_range):
        super(StructSemEmbDecoder, self).__init__()

        self.is_cuda_available = False
        if torch.cuda.is_available():
            print('CUDA/GPU device is available: [{}]'.format(torch.cuda.get_device_name(0)))
            self.is_cuda_available = True

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.item_num = item_num
        self.user_num = user_num
        self.vocob_size = vocab_size
        self.rating_range = rating_range

        self.user_embeddings = nn.Embedding(user_num, emb_dim)
        if self.is_cuda_available:
            self.user_embeddings = self.user_embeddings.cuda()

        self.dropout = nn.Dropout(0.5)

        self.hidden2rating = nn.Linear(self.emb_dim * 2, self.rating_range)

    def forward(self, combined_item_emb, user_id):
        user_emb = self.user_embeddings(torch.LongTensor([user_id])).view(self.emb_dim, 1, -1)

        combined_item_desc_emb = torch.cat([combined_item_emb, user_emb], 0)

        rating_space = self.hidden2rating(combined_item_desc_emb.view(1, -1, self.emb_dim * 2))
        rating_space = rating_space.view(1, self.rating_range)
        rating_score = F.softmax(rating_space, dim=1)
        return rating_score
