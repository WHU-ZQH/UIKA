import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.module.attention.dot_attention import DotAttention
from layers.squeeze_embedding import SqueezeEmbedding


class ABGCN(nn.Module):
    def __init__(self, args):
        super(ABGCN, self).__init__()
        self.args = args
        Ks = [2, 3, 4]
        if torch.cuda.is_available():
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float().cuda(), freeze=False)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float(), freeze=False)
        self.conv1 = nn.Conv1d(100, 100, kernel_size=4)
        self.convs1 = nn.ModuleList([nn.Conv1d(300, 100, K, dilation=1) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(300, 100, K, dilation=1) for K in Ks])

        self.convs3 = nn.ModuleList([nn.Conv1d(300, 100, K, dilation=1) for K in Ks])

        self.convs1.append(nn.Conv1d(300, 100, 3, dilation=2))
        self.convs2.append(nn.Conv1d(300, 100, 3, dilation=2))
        self.convs3.append(nn.Conv1d(300, 100, 3, dilation=2))

        self.convs4 = nn.ModuleList([nn.Conv1d(300, 100, K, padding=K - 2, dilation=1) for K in [3]])
        self.fc = nn.Linear(100, 3)
        self.fc_aspect = nn.Linear(100, 100)
        self.drop = nn.Dropout(0.2)
        self.attention = DotAttention()

    def location_feature(self, feature, offset):
        if torch.cuda.is_available():
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_().cuda()
        else:
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_()
        for i in range(offset.shape[0]):
            weight[i] = offset[i][:feature.shape[1]]
        feature = weight.unsqueeze(2) * feature
        return feature

    def forward(self, feature, aspect, offset):
        text, aspect, offset = feature.long(), aspect.long(), offset
        text_embed = self.embed(text)
        aspect_embed = self.embed(aspect)
        z1 = []
        aspect_v = [torch.relu(conv(aspect_embed.transpose(1, 2))) for conv in self.convs4]
        aspect_v = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aspect_v]
        aspect_v = torch.cat(aspect_v, 1)
        text_embed = self.location_feature(text_embed, offset)
        for conv in self.convs2:
            y1 = conv(text_embed.transpose(1, 2))
            out_at = self.attention(aspect_v.unsqueeze(1), y1.transpose(1, 2), y1.transpose(1, 2))
            z1.append(torch.relu(out_at.transpose(1, 2)))
        x = [torch.tanh(conv(text_embed.transpose(1, 2))) for conv in self.convs1]
        y = [torch.relu(conv(text_embed.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs3]
        z = [i * j + k for i, j, k in zip(x, y, z1)]
        z0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in z]
        z0 = [i.unsqueeze(2) for i in z0]
        z = torch.cat(z0, dim=2)
        z = self.conv1(z)
        z = z.view(z.shape[0], -1)
        z = self.drop(z)
        output = self.fc(z)
        return output


class ABGCN_pre(nn.Module):
    def __init__(self, args):
        super(ABGCN_pre, self).__init__()
        self.args = args
        Ks = [2, 3, 4]
        if torch.cuda.is_available():
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float().cuda(), freeze=False)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(args.embeddings).float(), freeze=False)
        self.conv1 = nn.Conv1d(100, 100, kernel_size=4)
        self.convs1 = nn.ModuleList([nn.Conv1d(300, 100, K, dilation=1) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(300, 100, K, dilation=1) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(300, 100, K, dilation=1) for K in Ks])

        self.convs1.append(nn.Conv1d(300, 100, 3, dilation=2))
        self.convs2.append(nn.Conv1d(300, 100, 3, dilation=2))
        self.convs3.append(nn.Conv1d(300, 100, 3, dilation=2))
        self.convs4 = nn.ModuleList([nn.Conv1d(300, 100, K, padding=K - 2, dilation=1) for K in [3]])
        if args.stage ==1 :
            self.fc2 = nn.Linear(100, 2)  # for first pretraining stage, dim= (100, 2), while second is (100, 3)
        elif args.stage ==2 :
            self.fc2 = nn.Linear(100, 3)
        self.fc_aspect = nn.Linear(100, 100)
        self.drop = nn.Dropout(0.2)
        self.attention = DotAttention()

    def forward(self, feature, aspect):
        text, aspect = feature.long(), aspect.long()
        text_embed = self.embed(text)
        aspect_embed = self.embed(aspect)
        z1 = []
        aspect_v = [torch.relu(conv(aspect_embed.transpose(1, 2))) for conv in self.convs4]
        aspect_v = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aspect_v]
        aspect_v = torch.cat(aspect_v, 1)
        for conv in self.convs2:
            y1 = conv(text_embed.transpose(1, 2))
            out_at = self.attention(aspect_v.unsqueeze(1), y1.transpose(1, 2), y1.transpose(1, 2))
            z1.append(F.relu(out_at.transpose(1, 2)))
        x = [torch.tanh(conv(text_embed.transpose(1, 2))) for conv in self.convs1]
        y = [torch.relu(conv(text_embed.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs3]
        z = [i * j + k for i, j, k in zip(x, y, z1)]
        z0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in z]
        z0 = [i.unsqueeze(2) for i in z0]
        z = torch.cat(z0, dim=2)
        z = self.conv1(z)
        z = z.view(z.shape[0], -1)
        z = self.drop(z)
        output = self.fc2(z)
        return output


class ABGCN_Bert(nn.Module):
    def __init__(self, bert):
        super(ABGCN_Bert, self).__init__()
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        Ks = [2, 3, 4]
        self.conv1_ = nn.Conv1d(100, 100, kernel_size=4)
        self.convs1_ = nn.ModuleList([nn.Conv1d(768, 100, K, dilation=1) for K in Ks])
        self.convs2_ = nn.ModuleList([nn.Conv1d(768, 100, K, dilation=1) for K in Ks])
        self.convs3_ = nn.ModuleList([nn.Conv1d(768, 100, K, dilation=1) for K in Ks])

        self.convs1_.append(nn.Conv1d(768, 100, 3, dilation=2))
        self.convs2_.append(nn.Conv1d(768, 100, 3, dilation=2))
        self.convs3_.append(nn.Conv1d(768, 100, 3, dilation=2))
        self.convs4_ = nn.ModuleList([nn.Conv1d(768, 100, K, padding=K - 2, dilation=1) for K in [3]])
        self.fc = nn.Linear(100, 3)
        self.fc_aspect_ = nn.Linear(100, 100)
        self.drop_ = nn.Dropout(0.2)
        self.drop_bert_ = nn.Dropout(0.2)
        self.attention = DotAttention()

    def location_feature(self, feature, offset):
        if torch.cuda.is_available():
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_().cuda()
        else:
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_()
        for i in range(offset.shape[0]):
            weight[i] = offset[i][:feature.shape[1]]
        feature = weight.unsqueeze(2) * feature
        return feature

    def forward(self, bert_token, bert_token_aspect, offset):
        bert_token, bert_token_aspect = bert_token.long(), bert_token_aspect.long()
        context_len = torch.sum(bert_token != 0, dim=-1).to("cpu")
        target_len = torch.sum(bert_token_aspect != 0, dim=-1).to("cpu")
        context = self.squeeze_embedding(bert_token, context_len)
        text_embed, _ = self.bert(context, output_all_encoded_layers=False)
        text_embed = self.drop_bert_(text_embed)
        target = self.squeeze_embedding(bert_token_aspect, target_len)
        aspect_embed, _ = self.bert(target, output_all_encoded_layers=False)
        aspect_embed = self.drop_bert_(aspect_embed)
        z1 = []
        aspect_v = [torch.relu(conv(aspect_embed.transpose(1, 2))) for conv in self.convs4_]
        aspect_v = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aspect_v]
        aspect_v = torch.cat(aspect_v, 1)
        text_embed = self.location_feature(text_embed, offset)
        for conv in self.convs2_:
            y1 = conv(text_embed.transpose(1, 2))
            out_at = self.attention(aspect_v.unsqueeze(1), y1.transpose(1, 2), y1.transpose(1, 2))
            z1.append(F.relu(out_at.transpose(1, 2)))
        x = [torch.tanh(conv(text_embed.transpose(1, 2))) for conv in self.convs1_]
        y = [torch.relu(conv(text_embed.transpose(1, 2)) + self.fc_aspect_(aspect_v).unsqueeze(2)) for conv in self.convs3_]
        z = [i * j + k for i, j, k in zip(x, y, z1)]
        z0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in z]
        z0 = [i.unsqueeze(2) for i in z0]
        z = torch.cat(z0, dim=2)
        z = self.conv1_(z)
        z = z.view(z.shape[0], -1)
        z = self.drop_(z)
        output = self.fc(z)
        return output


class ABGCN_pre_Bert(nn.Module):
    def __init__(self, bert):
        super(ABGCN_pre_Bert, self).__init__()
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        Ks = [2, 3, 4]
        self.conv1 = nn.Conv1d(100, 100, kernel_size=4)
        self.convs1 = nn.ModuleList([nn.Conv1d(768, 100, K, dilation=1) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(768, 100, K, dilation=1) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(768, 100, K, dilation=1) for K in Ks])
        self.convs1.append(nn.Conv1d(768, 100, 3, dilation=2))
        self.convs2.append(nn.Conv1d(768, 100, 3, dilation=2))
        self.convs3.append(nn.Conv1d(768, 100, 3, dilation=2))
        self.convs4 = nn.ModuleList([nn.Conv1d(768, 100, K, padding=K - 2, dilation=1) for K in [3]])
        self.fc2 = nn.Linear(100, 3)  # for first pretraining stage, dim= (100, 2), while second is (100, 3)
        self.fc_aspect = nn.Linear(100, 100)
        self.drop = nn.Dropout(0.2)
        self.drop_bert = nn.Dropout(0.2)
        self.attention = DotAttention()

    def forward(self, bert_token, bert_token_aspect):
        bert_token, bert_token_aspect = bert_token.long(), bert_token_aspect.long()
        context_len = torch.sum(bert_token != 0, dim=-1).to("cpu")
        target_len = torch.sum(bert_token_aspect != 0, dim=-1).to("cpu")
        context = self.squeeze_embedding(bert_token, context_len)
        text_embed, _ = self.bert(context, output_all_encoded_layers=False)
        text_embed = self.drop_bert(text_embed)
        target = self.squeeze_embedding(bert_token_aspect, target_len)
        aspect_embed, _ = self.bert(target, output_all_encoded_layers=False)
        aspect_embed = self.drop_bert(aspect_embed)
        z1 = []
        aspect_v = [F.relu(conv(aspect_embed.transpose(1, 2))) for conv in self.convs4]
        aspect_v = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aspect_v]
        aspect_v = torch.cat(aspect_v, 1)
        for conv in self.convs2:
            y1 = conv(text_embed.transpose(1, 2))
            out_at = self.attention(aspect_v.unsqueeze(1), y1.transpose(1, 2), y1.transpose(1, 2))
            z1.append(F.relu(out_at.transpose(1, 2)))
        x = [F.tanh(conv(text_embed.transpose(1, 2))) for conv in self.convs1]
        y = [F.relu(conv(text_embed.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs3]
        z = [i * j + k for i, j, k in zip(x, y, z1)]
        z0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in z]
        z0 = [i.unsqueeze(2) for i in z0]
        z = torch.cat(z0, dim=2)
        z = self.conv1(z)
        z = z.view(z.shape[0], -1)
        z = self.drop(z)
        output = self.fc2(z)
        return output
