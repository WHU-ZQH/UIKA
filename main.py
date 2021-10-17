# -*- coding: utf-8 -*-
import argparse
import time
import os
from model.ABGCN import ABGCN, ABGCN_pre, ABGCN_Bert, ABGCN_pre_Bert
from model.ATAE_LSTM import ATAE_LSTM, ATAE_LSTM_pre, ATAE_LSTM_Bert, ATAE_LSTM_pre_Bert
from model.GCAE import GCAE, GCAE_pre, GCAE_Bert, GCAE_pre_Bert
from utils import *
from mydataset import *
import numpy as np
import random
import torch.nn.functional as F
from sklearn import metrics

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
seed = 14
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train_pre(args):
    print('This is first pre_training:')
    dataset, embeddings = build_dataset(ds_name=args.pre_name, bs=args.bs, train_pre=True)
    model_path = 'stages_saved_model/first_stage/%s_Amazon_review_pre.pth'%args.model
    args.embeddings = embeddings
    train_set, test_set = dataset
    input_list = ['wids', 'tids', 'y']
    trainset, testset = my_dataset(train_set, input_list), my_dataset(test_set, input_list)
    train_loader, test_loader = DataLoader(dataset=trainset, batch_size=args.bs, shuffle=True, num_workers=2), \
                                DataLoader(dataset=testset, batch_size=args.bs, shuffle=True, num_workers=2)
    if args.model == 'ABGCN':
        model = ABGCN_pre(args=args)
    elif args.model == 'GCAE':
        model = GCAE_pre(args=args)
    elif args.model == 'ATAE':
        model = ATAE_LSTM_pre(args=args)
    else:
        print('model error')
    if torch.cuda.is_available():
        model = model.cuda()

    max_acc, max_f1, test_acc, test_f1 = 0, 0, 0, 0
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    model.train()
    for i in range(1, 6):
        for j, input in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            train_x, train_xt, train_y = input
            if torch.cuda.is_available():
                train_x, train_xt, train_y = train_x.cuda(), train_xt.cuda(), train_y.cuda()
            logit = model(train_x, train_xt)
            loss = F.cross_entropy(logit, train_y)
            loss.backward()
            optimizer.step()
            corrects = (torch.max(logit, 1)[1].view(train_y.size()).data == train_y.data).sum()
            accuracy = 100.0 * corrects / train_y.shape[0]
            f1 = metrics.f1_score(train_y.cpu(), torch.argmax(logit, -1).cpu(), labels=[0, 1, 2], average='macro')
            if j % 20 == 0:
                test_acc, test_f1 = eval_pre(model, test_loader)
                if max_acc < test_acc:
                    max_acc = test_acc
                    model_dict = model.state_dict()
                    del model_dict['embed.weight']
                    torch.save(model_dict, model_path)
                if max_f1 < test_f1:
                    max_f1 = test_f1
                print(
                    '\r - loss: {:.6f} f1:{:.4f} acc: {:.4f}%({}/{})'.format(loss.item(), f1, accuracy, corrects,
                                                                             train_y.shape[0]))
        print("In Epoch %s: test_accuracy: %.2f, test_macro-f1: %.2f\n" % (i, test_acc * 100, test_f1 * 100))
    return model_path


def train_pre_bert(args):
    print('This is pre_bert_training:')
    dataset = build_dataset(ds_name=args.pre_name, bs=args.bs, train_pre=True, is_bert=True)
    model_path = 'stages_saved_model/first_stage/%s_Amazon_review_bert_pre.pth'%args.model
    train_set, test_set = dataset
    input_list = ['bert_token', 'bert_token_aspect', 'y']
    trainset, testset = my_dataset(train_set, input_list), my_dataset(test_set, input_list)
    train_loader, test_loader = DataLoader(dataset=trainset, batch_size=args.bs, shuffle=True, num_workers=2), \
                                DataLoader(dataset=testset, batch_size=args.bs, shuffle=True, num_workers=2)
    bert = BertModel.from_pretrained(r'D:\BERT\bert-base-uncased')  ## the path of your bertmodel
    if args.model == 'ABGCN':
        model = ABGCN_pre_Bert(bert=bert)
    elif args.model == 'GCAE':
        model = GCAE_pre_Bert(bert=bert)
    elif args.model == 'ATAE':
        model = ATAE_LSTM_pre_Bert(bert=bert)
    else:
        print('model error')
    if torch.cuda.is_available():
        model = model.cuda()

    max_acc, max_f1, test_acc, test_f1 = 0, 0, 0, 0
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    model.train()
    for i in range(1, args.n_epoch + 1):
        for j, input in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            train_bert_token, train_bert_token_aspect, train_y = input
            if torch.cuda.is_available():
                train_bert_token, train_bert_token_aspect, train_y = train_bert_token.cuda(), train_bert_token_aspect.cuda(), train_y.cuda()
            logit = model(train_bert_token, train_bert_token_aspect)
            loss = F.cross_entropy(logit, train_y)
            loss.backward()
            optimizer.step()
            corrects = (torch.max(logit, 1)[1].view(train_y.size()).data == train_y.data).sum()
            accuracy = 100.0 * corrects / train_y.shape[0]
            f1 = metrics.f1_score(train_y.cpu(), torch.argmax(logit, -1).cpu(), labels=[0, 1, 2], average='macro')
            if j % 20 == 0:
                test_acc, test_f1 = eval_pre_bert(model, test_loader)
                if max_acc < test_acc:
                    max_acc = test_acc
                    model_dict = model.state_dict()
                    torch.save(model_dict, model_path)
                if max_f1 < test_f1:
                    max_f1 = test_f1
                print(
                    '\r - loss: {:.6f} f1:{:.4f} acc: {:.4f}%({}/{})'.format(loss.item(), f1, accuracy, corrects,
                                                                             train_y.shape[0]))
        print('\nEvaluation - acc: {:.4f} f1: {:.4f} '.format(max_acc, max_f1))
    return model_path


def train_again(args, pre_path):
    print('This is second pre_training:')
    model_path = 'stages_saved_model/second_stage/Amazon_%s_learner.pth'%args.model
    dataset, embeddings = build_dataset(ds_name=args.ds_name, bs=args.bs)
    args.embeddings = embeddings
    train_set, test_set = dataset
    input_list = ['wids', 'tids', 'y']
    trainset, testset = my_dataset(train_set, input_list), my_dataset(test_set, input_list)
    train_loader, test_loader = DataLoader(dataset=trainset, batch_size=args.bs, shuffle=True, num_workers=2), \
                                DataLoader(dataset=testset, batch_size=args.bs, shuffle=True, num_workers=2)

    if args.model == 'ABGCN':
        model_guidance = ABGCN_pre(args=args)
        model_learner = ABGCN_pre(args=args)
    elif args.model == 'GCAE':
        model_guidance = GCAE_pre(args=args)
        model_learner= GCAE_pre(args=args)
    elif args.model == 'ATAE':
        model_guidance = ATAE_LSTM_pre(args=args)
        model_learner = ATAE_LSTM_pre(args=args)
    else:
        print('model error')
    if torch.cuda.is_available():
        model_guidance = model_guidance.cuda()
        model_learner = model_learner.cuda()
    pretrained_dict = torch.load(pre_path, map_location='cuda')
    model_dict_guidance = model_guidance.state_dict()
    initialized_dict_guidance = {k: v for k, v in pretrained_dict.items() if k in model_dict_guidance}
    model_dict_guidance.update(initialized_dict_guidance)
    model_guidance.load_state_dict(model_dict_guidance)

    model_dict_learner = model_learner.state_dict()
    initialized_dict_learner = {k: v for k, v in pretrained_dict.items() if k in model_dict_learner}
    model_dict_learner.update(initialized_dict_learner)
    model_learner.load_state_dict(model_dict_learner)
    max_acc, max_f1 = 0, 0
    optimizer_student = torch.optim.Adagrad(model_guidance.parameters(), lr=args.learning_rate)
    model_guidance.train()
    for i in range(1, args.n_epoch + 1):  ##args.n_epoch + 1
        for j, input in enumerate(train_loader):
            model_guidance.train()
            train_x, train_xt, train_y = input
            if torch.cuda.is_available():
                train_x, train_xt, train_y = train_x.cuda(), train_xt.cuda(), train_y.cuda()
            model_dict_guidance = model_guidance.state_dict().copy()
            model_dict_learner = model_learner.state_dict().copy()
            model_dict_guidance = {k: v for k, v in model_dict_guidance.items()}
            model_dict_learner = {k: v for k, v in model_dict_learner.items()}
            model_dict = {}
            for k in model_dict_learner.keys():
                parameters = 0.01 * model_dict_guidance.get(k) + 0.99 * model_dict_learner.get(k)
                model_dict[k] = parameters
            model_learner.load_state_dict(model_dict)
            for name, param in model_learner.named_parameters():
                param.requires_grad = False
            logit = model_guidance(train_x, train_xt)
            logit2 = model_learner(train_x, train_xt)
            logit.requires_grad_()
            logit2.requires_grad_()
            loss = (1-args.alpha) * F.mse_loss(logit, logit2) + args.alpha * F.cross_entropy(logit, train_y)
            loss.backward()
            optimizer_student.step()
            corrects = (torch.max(logit, 1)[1].view(train_y.size()).data == train_y.data).sum()
            accuracy = 100.0 * corrects / train_y.shape[0]
            f1 = metrics.f1_score(train_y.cpu(), torch.argmax(logit, -1).cpu(), labels=[0, 1, 2], average='macro')
            if j % 10 == 0:
                test_acc, test_f1 = eval_pre(model_learner, test_loader)
                if max_acc < test_acc:
                    max_acc = test_acc
                    model_dict = model_learner.state_dict()
                    del model_dict['embed.weight']
                    torch.save(model_dict, model_path)
                if max_f1 < test_f1:
                    max_f1 = test_f1
                print(
                    '\r - loss_learner: {:.6f} f1:{:.4f} acc: {:.4f}%({}/{})'.format(loss.item(), f1, accuracy,
                                                                                     corrects, train_y.shape[0]))
        print('\nEvaluation - acc: {:.4f} f1: {:.4f} '.format(max_acc, max_f1))
    return model_path


def train_again_bert(args, pre_path):
    print('This is second pre_training_bert:')
    model_path = 'stages_saved_model/second_stage/Amazon_%s_learner_bert.pth'%args.model
    dataset = build_dataset(ds_name=args.ds_name, bs=args.bs, is_bert=True)
    train_set, test_set = dataset
    input_list = ['bert_token', 'bert_token_aspect', 'y']
    trainset, testset = my_dataset(train_set, input_list), my_dataset(test_set, input_list)
    train_loader, test_loader = DataLoader(dataset=trainset, batch_size=args.bs, shuffle=True, num_workers=2), \
                                DataLoader(dataset=testset, batch_size=args.bs, shuffle=True, num_workers=2)

    bert = BertModel.from_pretrained(r'D:\BERT\bert-base-uncased')  ##the path of your bert model
    if args.model == 'ABGCN':
        model_guidance = ABGCN_pre_Bert(bert=bert)
        model_learner = ABGCN_pre_Bert(bert=bert)
    elif args.model == 'GCAE':
        model_guidance = GCAE_pre_Bert(bert=bert)
        model_learner= GCAE_pre_Bert(bert=bert)
    elif args.model == 'ATAE':
        model_guidance = ATAE_LSTM_pre_Bert(bert=bert)
        model_learner = ATAE_LSTM_pre_Bert(bert=bert)
    else:
        print('model error')
    
    if torch.cuda.is_available():
        model_guidance = model_guidance.cuda()
        model_learner = model_learner.cuda()
    pretrained_dict = torch.load(pre_path, map_location='cuda')
    model_dict_guidance = model_guidance.state_dict()
    initialized_dict_guidance = {k: v for k, v in pretrained_dict.items() if k in model_dict_guidance}
    model_dict_guidance.update(initialized_dict_guidance)
    model_guidance.load_state_dict(model_dict_guidance)

    model_dict_learner = model_learner.state_dict()
    initialized_dict_learner = {k: v for k, v in pretrained_dict.items() if k in model_dict_learner}
    model_dict_learner.update(initialized_dict_learner)
    model_learner.load_state_dict(model_dict_learner)

    max_acc, max_f1 = 0, 0
    optimizer_student = torch.optim.Adagrad(model_guidance.parameters(), lr=0.00001)
    model_guidance.train()
    for i in range(1, args.n_epoch + 1):  ##args.n_epoch + 1
        for j, input in enumerate(train_loader):
            model_guidance.train()
            train_bert_token, train_bert_token_aspect, train_y, train_pw = input
            if torch.cuda.is_available():
                train_bert_token, train_bert_token_aspect, train_y, train_pw = train_bert_token.cuda(), train_bert_token_aspect.cuda(), train_y.cuda(), train_pw.cuda()
            model_dict_guidance = model_guidance.state_dict().copy()
            model_dict_learner = model_learner.state_dict().copy()
            model_dict_guidance = {k: v for k, v in model_dict_guidance.items()}
            model_dict_learner = {k: v for k, v in model_dict_learner.items()}
            model_dict = {}
            for k in model_dict_learner.keys():
                parameters = 0.01 * model_dict_guidance.get(k) + 0.99 * model_dict_learner.get(k)
                model_dict[k] = parameters
            model_learner.load_state_dict(model_dict)
            for name, param in model_learner.named_parameters():
                param.requires_grad = False
            logit = model_guidance(train_bert_token, train_bert_token_aspect)
            logit2 = model_learner(train_bert_token, train_bert_token_aspect)
            logit.requires_grad_()
            logit2.requires_grad_()
            loss = (1-args.alpha) * F.mse_loss(logit, logit2) + args.alpha * F.cross_entropy(logit, train_y)
            loss.backward()
            optimizer_student.step()
            corrects = (torch.max(logit, 1)[1].view(train_y.size()).data == train_y.data).sum()
            accuracy = 100.0 * corrects / train_y.shape[0]
            f1 = metrics.f1_score(train_y.cpu(), torch.argmax(logit, -1).cpu(), labels=[0, 1, 2], average='macro')
            if j % 10 == 0:
                test_acc, test_f1 = eval_pre_bert(model_learner, test_loader)
                if max_acc < test_acc:
                    max_acc = test_acc
                    model_dict = model_learner.state_dict()
                    torch.save(model_dict, model_path)
                if max_f1 < test_f1:
                    max_f1 = test_f1
                print(
                    '\r - loss_learner: {:.6f} f1:{:.4f} acc: {:.4f}%({}/{})'.format(loss.item(), f1, accuracy,
                                                                                     corrects, train_y.shape[0]))
        print('\nEvaluation - acc: {:.4f} f1: {:.4f} '.format(max_acc, max_f1))
    return model_path


def eval_pre(model, test_loader):
    model.eval()
    t_targets_all, t_outputs_all = None, None
    with torch.no_grad():
        corrects, f1, avg_loss, size = 0, 0, 0, 0
        for j, input in enumerate(test_loader):
            test_x, test_xt, test_y = input
            if torch.cuda.is_available():
                test_x, test_xt, test_y = test_x.cuda(), test_xt.cuda(), test_y.cuda()
            logit = model(test_x, test_xt)
            loss = F.cross_entropy(logit, test_y, reduction='sum')
            avg_loss += loss.item()
            size += test_y.size(0)
            corrects += (torch.max(logit, 1)
                         [1].view(test_y.size()).data == test_y.data).sum()
            if t_targets_all is None:
                t_targets_all = test_y
                t_outputs_all = logit
            else:
                t_targets_all = torch.cat((t_targets_all, test_y), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, logit), dim=0)
        accuracy = 1.0 * corrects / size
        F1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1,2],
                              average='macro')
    return accuracy, F1


def eval_pre_bert(model, test_loader):
    model.eval()
    t_targets_all, t_outputs_all = None, None
    with torch.no_grad():
        corrects, f1, avg_loss, size = 0, 0, 0, 0
        for j, input in enumerate(test_loader):
            train_bert_token, train_bert_token_aspect, test_y = input
            if torch.cuda.is_available():
                train_bert_token, train_bert_token_aspect, test_y = train_bert_token.cuda(), train_bert_token_aspect.cuda(), test_y.cuda()
            logit = model(train_bert_token, train_bert_token_aspect)
            loss = F.cross_entropy(logit, test_y, reduction='sum')
            avg_loss += loss.item()
            size += test_y.size(0)
            corrects += (torch.max(logit, 1)
                         [1].view(test_y.size()).data == test_y.data).sum()
            if t_targets_all is None:
                t_targets_all = test_y
                t_outputs_all = logit
            else:
                t_targets_all = torch.cat((t_targets_all, test_y), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, logit), dim=0)
        accuracy = 1.0 * corrects / size
        F1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1,2],
                              average='macro')
    return accuracy, F1


def train(args, path, is_save=False):
    dataset, embeddings = build_dataset(ds_name=args.ds_name, bs=args.bs)
    args.embeddings = embeddings
    train_set, test_set = dataset
    input_list = ['wids', 'tids', 'y', 'pw']
    trainset, testset = my_dataset(train_set, input_list), my_dataset(test_set, input_list)
    train_loader, test_loader = DataLoader(dataset=trainset, batch_size=args.bs, shuffle=True, num_workers=0), \
                                DataLoader(dataset=testset, batch_size=args.bs, shuffle=True, num_workers=0)
    if args.model == 'ABGCN':
        model = ABGCN(args=args).cuda()
    elif args.model == 'GCAE':
        model = GCAE(args=args)
    elif args.model == 'ATAE':
        model = ATAE_LSTM(args=args)
    else:
        print('model error')
    if torch.cuda.is_available():
        model = model.cuda()
    load_path = path
    save_path = 'stages_saved_model/third_stage/{}_third_{}.pth'.format(args.model,args.ds_name)
    if args.stage != 4:
        pretrained_dict = torch.load(load_path, map_location='cuda')
        model_dict = model.state_dict()
        initialized_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(initialized_dict)
        model.load_state_dict(model_dict)
    train_time = []
    result_store_test = [[], []]
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    model.train()
    for i in range(1, args.n_epoch + 1):
        beg = time.time()
        max_acc, max_f1 = 0, 0
        for j, input in enumerate(train_loader):
            model.train()
            train_x, train_xt, train_y, train_pw = input
            if torch.cuda.is_available():
                train_x, train_xt, train_y, train_pw = train_x.cuda(), train_xt.cuda(), train_y.cuda(), train_pw.cuda()
            logit = model(train_x, train_xt, train_pw)
            loss = F.cross_entropy(logit, train_y)
            loss.backward()
            optimizer.step()
            corrects = (torch.max(logit, 1)[1].view(train_y.size()).data == train_y.data).sum()
            accuracy = 100.0 * corrects / train_y.shape[0]
            f1 = metrics.f1_score(train_y.cpu(), torch.argmax(logit, -1).cpu(), labels=[0, 1, 2], average='macro')
            if j % 10 == 0:
                test_acc, test_f1 = eval(model, test_loader)
                if max_acc < test_acc:
                    max_acc = test_acc
                    if is_save:
                        torch.save(model.module.state_dict(), save_path)
                if max_f1 < test_f1:
                    max_f1 = test_f1
                print(
                    '\r - loss: {:.6f} f1:{:.4f} acc: {:.4f}%({}/{})'.format(loss.item(), f1, accuracy, corrects,
                                                                             train_y.shape[0]))
        #print('\nEvaluation - acc: {:.4f} f1: {:.4f} '.format(max_acc, max_f1))
        end = time.time()
        train_time.append(end - beg)
        result_store_test[0].append(max_acc)
        result_store_test[1].append(max_f1)
        print("In Epoch %s: test_accuracy: %.2f, test_macro-f1: %.2f\n" % (i, max_acc * 100, max_f1 * 100))
    avg_time = sum(train_time) / len(train_time)
    best_index_acc = result_store_test[0].index(max(result_store_test[0]))
    print("Best model in Epoch %s: test accuracy: %.2f, macro-f1: %.2f ,avg_time: %.2f\n" % (
        best_index_acc + 1, max(result_store_test[0]), max(result_store_test[1]), avg_time))
    return max(result_store_test[0]), max(result_store_test[1]), avg_time


def train_bert(args, path, is_save=False):
    dataset = build_dataset(ds_name=args.ds_name, bs=args.bs, train_pre=False, is_bert=True)
    train_set, test_set = dataset
    input_list = ['bert_token', 'bert_token_aspect', 'y', 'pw']
    trainset, testset = my_dataset(train_set, input_list), my_dataset(test_set, input_list)
    train_loader, test_loader = DataLoader(dataset=trainset, batch_size=args.bs, shuffle=True, num_workers=2), \
                                DataLoader(dataset=testset, batch_size=args.bs, shuffle=True, num_workers=2)
    bert = BertModel.from_pretrained(r'D:\BERT\bert-base-uncased')  ## the path of your bertmodel
    
    if args.model == 'ABGCN':
        model = ABGCN_Bert(bert=bert)
    elif args.model == 'GCAE':
        model = GCAE_Bert(bert=bert)
    elif args.model == 'ATAE':
        model = ATAE_LSTM_Bert(bert=bert)
    else:
        print('model error')
    if torch.cuda.is_available():
        model = model.cuda()
    load_path = path
    save_path = 'stages_saved_model/third_stage/{}_third_{}_bert.pth'.format(args.model,args.ds_name)
    if args.stage != 4:
        pretrained_dict = torch.load(load_path, map_location='cuda')
        model_dict = model.state_dict()
        initialized_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(initialized_dict)
        model.load_state_dict(model_dict)
    model = torch.nn.DataParallel(model, device_ids=[0])
    train_time = []
    result_store_test = [[], []]
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.module.parameters()), lr=args.learning_rate)
    model.train()
    for i in range(1, args.n_epoch + 1):
        beg = time.time()
        max_acc, max_f1 = 0, 0
        for j, input in enumerate(train_loader):
            model.train()
            train_bert_token, train_bert_token_aspect, train_y, train_pw = input
            if torch.cuda.is_available():
                train_bert_token, train_bert_token_aspect, train_y, train_pw = train_bert_token.cuda(), train_bert_token_aspect.cuda(), train_y.cuda(), train_pw.cuda()
            logit = model(train_bert_token, train_bert_token_aspect, train_pw)
            loss = F.cross_entropy(logit, train_y)
            loss.backward()
            optimizer.step()
            corrects = (torch.max(logit, 1)[1].view(train_y.size()).data == train_y.data).sum()
            accuracy = 100.0 * corrects / train_y.shape[0]
            f1 = metrics.f1_score(train_y.cpu(), torch.argmax(logit, -1).cpu(), labels=[0, 1, 2], average='macro')
            if j % 10 == 0:
                print(
                    '\r - loss: {:.6f} f1:{:.4f} acc: {:.4f}%({}/{})'.format(loss.item(), f1, accuracy, corrects,
                                                                             train_y.shape[0]))
        test_acc, test_f1 = eval_bert(model, test_loader)
        if max_acc < test_acc:
            max_acc = test_acc
            if is_save:
                torch.save(model.module.state_dict(), save_path)
        if max_f1 < test_f1:
            max_f1 = test_f1
        print('\nEvaluation - acc: {:.4f} f1: {:.4f} '.format(max_acc, max_f1))
        end = time.time()
        train_time.append(end - beg)
        result_store_test[0].append(max_acc)
        result_store_test[1].append(max_f1)
        print("In Epoch %s: test_accuracy: %.2f, test_macro-f1: %.2f\n" % (i, test_acc * 100, test_f1 * 100))
    avg_time = sum(train_time) / len(train_time)
    best_index_acc = result_store_test[0].index(max(result_store_test[0]))
    print("Best model in Epoch %s: test accuracy: %.2f, macro-f1: %.2f ,avg_time: %.2f\n" % (
        best_index_acc + 1, max(result_store_test[0]), max(result_store_test[1]), avg_time))
    return max(result_store_test[0]), max(result_store_test[1]), avg_time


def eval(model, test_loader):
    model.eval()
    t_targets_all, t_outputs_all = None, None
    with torch.no_grad():
        corrects, f1, avg_loss, size = 0, 0, 0, 0
        loss = None
        for j, input in enumerate(test_loader):
            test_x, test_xt, test_y, test_pw = input
            if torch.cuda.is_available():
                test_x, test_xt, test_y, test_pw = test_x.cuda(), test_xt.cuda(), test_y.cuda(), test_pw.cuda()
            logit = model(test_x, test_xt, test_pw)
            loss = F.cross_entropy(logit, test_y, reduction='sum')
            avg_loss += loss.item()
            size += test_y.size(0)
            corrects += (torch.max(logit, 1)
                         [1].view(test_y.size()).data == test_y.data).sum()
            if t_targets_all is None:
                t_targets_all = test_y
                t_outputs_all = logit
            else:
                t_targets_all = torch.cat((t_targets_all, test_y), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, logit), dim=0)
        accuracy = 1.0 * corrects / size
        F1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
    return accuracy, F1


def eval_bert(model, test_loader):
    model.eval()
    t_targets_all, t_outputs_all = None, None
    with torch.no_grad():
        corrects, f1, avg_loss, size = 0, 0, 0, 0
        for j, input in enumerate(test_loader):
            train_bert_token, train_bert_token_aspect, test_y, train_pw = input
            if torch.cuda.is_available():
                train_bert_token, train_bert_token_aspect, test_y, train_pw = train_bert_token.cuda(), train_bert_token_aspect.cuda(), test_y.cuda(), train_pw.cuda()
            logit = model(train_bert_token, train_bert_token_aspect, train_pw)
            loss = F.cross_entropy(logit, test_y, reduction='sum')
            avg_loss += loss.item()
            size += test_y.size(0)
            corrects += (torch.max(logit, 1)
                         [1].view(test_y.size()).data == test_y.data).sum()
            if t_targets_all is None:
                t_targets_all = test_y
                t_outputs_all = logit
            else:
                t_targets_all = torch.cat((t_targets_all, test_y), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, logit), dim=0)
        accuracy = 1.0 * corrects / size
        F1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
    return accuracy, F1


def evaluate_model(args, model_path):
    if args.is_bert == 1:
        dataset = build_dataset(ds_name=args.ds_name,
                                bs=args.bs, is_bert=True)
        train_set, test_set = dataset
        input_list = ['bert_token', 'bert_token_aspect', 'y', 'pw']
        testset = my_dataset(test_set, input_list)
        test_loader = DataLoader(dataset=testset, batch_size=args.bs, shuffle=True, num_workers=1)
        bert = BertModel.from_pretrained(r' ')  ##bert model path
        model = ABGCN_Bert(bert=bert).cuda()
        model.load_state_dict(torch.load(model_path))
        test_acc, test_f1 = eval_bert(model, test_loader)
    else:
        dataset, embeddings = build_dataset(ds_name=args.ds_name, bs=args.bs)
        args.embeddings = embeddings
        train_set, test_set = dataset
        input_list = ['wids', 'tids', 'y', 'pw']
        testset = my_dataset(test_set, input_list)
        test_loader = DataLoader(dataset=testset, batch_size=args.bs, shuffle=True, num_workers=1)
        if args.model == 'ABGCN':
            model = ABGCN(args=args).cuda()
        elif args.model == 'GCAE':
            model = GCAE(args=args).cuda()
        elif args.model == 'ATAE':
            model = ATAE_LSTM(args=args).cuda()
        else:
            print('model error')
        model.load_state_dict(torch.load(model_path, map_location='cuda'))   ##cuda is avilable--cuda, else cpu
        test_acc, test_f1 = eval(model, test_loader)
    return test_acc, test_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KGP settings')
    parser.add_argument("-ds_name", type=str, default="14semeval_rest",
                        help="dataset name")  # 14semeval_rest, 14semeval_laptop, Twitter
    parser.add_argument("-pre_name", type=str, default="Amazon",
                        help="pretraining dataset name")  # Amazon, Yelp
    parser.add_argument("-bs", type=int, default=64, help="batch size")
    parser.add_argument("-learning_rate", type=float, default=0.001, help="learning rate for sentimental features,  0.00003 for bert model")
    parser.add_argument("-n_epoch", type=int, default=10, help="number of training epoch")
    parser.add_argument("-model", type=str, default='ABGCN', help="model name")
    parser.add_argument("-is_test", type=int, default=1, help="test the model: 1 for test")
    parser.add_argument("-is_bert", type=int, default=0, help="glove-based model: 1 for bert")
    parser.add_argument("-alpha", type=float, default=0.6, help="weighting factor to control the knowledge transferring")
    parser.add_argument("-stage", type=int, default=4,
                        help="1 for first stage, 2 for second stage, 3 for third stage, 4 for training from scratch")
    args = parser.parse_args()
    acc, f1 = [], []
    acc2, f1_2 = [], []
    train_time = []
    path = ''
    if args.is_test == 1:
        test_path = "best_model_weight/{}_{}.pth".format(args.model,args.ds_name)  ## the path of test-model-weight
        test_acc, test_f1 = evaluate_model(args, test_path)
        print("Test : acc: {} f1: {}".format(test_acc, test_f1))
    else:
        if args.is_bert == 1:
            if args.stage == 1:
                path = train_pre_bert(args)
                print('The pretraining model in first stage is saved in {}'.format(path))
            elif args.stage == 2:
                path1 = 'stages_saved_model/first_stage/%s_Amazon_review_bert_pre.pth'%args.model ## the path of first pretraining stage
                path2 = train_again_bert(args, path1)
                print('The pretraining model in second stage is saved in {}'.format(path2))
            elif args.stage == 3:
                path = 'stages_saved_model/second_stage/Amazon_%s_learner.pth'%args.model  ## the path of second pretraining stage
        else:
            if args.stage == 1:
                path = train_pre(args)
                print('The pretraining model in first stage is saved in {}'.format(path))
            elif args.stage == 2:
                path1 = "stages_saved_model/first_stage/%s_Amazon_review_pre.pth"%args.model  ## the path of first pretraining stage
                path2 = train_again(args, path1)
                print('The pretraining model in second stage is saved in {}'.format(path2))
            elif args.stage == 3:
                path = 'stages_saved_model/second_stage/Amazon_%s_learner.pth'%args.model  ## the path of second pretraining stage
        if args.stage > 2:
            save_iteration = None
            for i in range(5):
                if (i + 1) == save_iteration:
                    if args.is_bert == 1:
                        a_acc, a_f1, a_time = train_bert(args, path, is_save=True)
                    else:
                        a_acc, a_f1, a_time = train(args, path, is_save=True)
                else:
                    if args.is_bert == 1:
                        a_acc, a_f1, a_time = train_bert(args, path, is_save=False)
                    else:
                        a_acc, a_f1, a_time = train(args, path, is_save=False)
                acc.append(a_acc)
                f1.append(a_f1)
                train_time.append(a_time)

            best_acc = max(acc)
            best_f1 = max(f1)
            avg_acc = sum(acc) / len(acc)
            avg_f1 = sum(f1) / len(f1)
            best_time = min(train_time)
            avg_time = sum(train_time) / len(train_time)
            print('The results of {} : '.format(args.ds_name), '\n',
                  'best_acc: {}  best_f1: {} min_time: {}'.format(best_acc, best_f1, best_time), '\n',
                  'avg_acc: {}  avg_f1: {} avg_time: {}'.format(avg_acc, avg_f1, avg_time))
        else:
            print('Done..., waiting for next stage')
