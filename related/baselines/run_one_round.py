import os
import time
import torch
from torchvision import models, datasets, transforms
from nasnet.nasnet import NASNetALarge
import bert.modeling as modeling
from bert.optimization import BertAdam
import utils
import numpy as np

from multiprocessing import Process, Barrier
import multiprocessing as mp

def nasnet_worker(barrier, quota):
    durations = []
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(quota)
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")
    model = NASNetALarge(num_classes=1000)
    model = model.to(device)
    model.train()
    metric_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


    img_size = 331
    batch_size = 4
    train_loader = utils.DummyDataLoader(batch=(
        torch.rand([batch_size, 3, img_size, img_size]),
        torch.ones([batch_size]).to(torch.long)
    ))


    for batch_idx, batch in enumerate(train_loader):
        if batch_idx > 100:
            break
        data, target = batch[0].to(device), batch[1].to(device)
        torch.cuda.synchronize()

        # cuda_graph = torch.cuda.CUDAGraph()
        # with torch.cuda.graph(cuda_graph):
        #     output = model(data)
        # cuda_graph.replay()

        barrier.wait()
        start = time.time()
        output = model(data)
        loss = metric_fn(output, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        duration = time.time() - start
        if batch_idx > 20:
            durations.append(duration)

    print("Nasnet duration: {:.4f} s".format(np.mean(durations)))

def resnet50_worker(barrier, quota):

    durations = []
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(quota)
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")

    model = models.__dict__['resnet50'](num_classes=1000)
    model = model.to(device)
    model.train()
    metric_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    img_size = 224
    batch_size = 8
    train_loader = utils.DummyDataLoader(batch=(
        torch.rand([batch_size, 3, img_size, img_size]),
        torch.ones([batch_size]).to(torch.long)
    ))

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx > 1000:
            break
        barrier.wait()
        start = time.time()

        data, target = batch[0].to(device), batch[1].to(device)
        output = model(data)
        loss = metric_fn(output, target)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        duration = time.time() - start
        if batch_idx > 100:
            durations.append(duration)

        # print("resnet one epoch -----")
    
    print("ResNet50 duration: {:.4f} s".format(np.mean(durations)))


def resnet101_worker(barrier, quota):

    durations = []
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(quota)
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")

    model = models.__dict__['resnet101'](num_classes=1000)
    model = model.to(device)
    model.train()
    metric_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    img_size = 224
    batch_size = 8
    train_loader = utils.DummyDataLoader(batch=(
        torch.rand([batch_size, 3, img_size, img_size]),
        torch.ones([batch_size]).to(torch.long)
    ))

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx > 1000:
            break
        barrier.wait()
        start = time.time()

        data, target = batch[0].to(device), batch[1].to(device)
        output = model(data)
        loss = metric_fn(output, target)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        duration = time.time() - start
        if batch_idx > 100:
            durations.append(duration)

    print("ResNet101 duration: {:.4f} s".format(np.mean(durations)))

def bert_worker(barrier, quota):
    durations = []
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(quota)
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")

    config_dict ={
          "attention_probs_dropout_prob": 0.1,
          "hidden_act": "gelu",
          "hidden_dropout_prob": 0.1,
          "hidden_size": 768,
          "initializer_range": 0.02,
          "intermediate_size": 3072,
          "max_position_embeddings": 512,
          "num_attention_heads": 12,
          "num_hidden_layers": 12,
          "type_vocab_size": 2,
          "vocab_size": 30522
        }
    
    config = modeling.BertConfig.from_dict(config_dict)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = modeling.BertForQuestionAnswering(config).to(device)

    batch_size = 8
    pin_memory = True
    input_ids = torch.ones((batch_size, 384), pin_memory=pin_memory).to(torch.int64)
    segment_ids = torch.ones((batch_size, 384), pin_memory=pin_memory).to(torch.int64)
    input_mask = torch.ones((batch_size, 384), pin_memory=pin_memory).to(torch.int64)
    start_positions = torch.zeros((batch_size, ), pin_memory=pin_memory).to(torch.int64)
    end_positions = torch.ones((batch_size, ), pin_memory=pin_memory).to(torch.int64)
    dataloader = utils.DummyDataLoader(batch=(input_ids, input_mask, segment_ids, start_positions, end_positions))

    param_optimizer = list(model.named_parameters())
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1, t_total=100)
    model.train()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx > 1000:
            break
        barrier.wait()
        start = time.time()

        input_ids, input_mask, segment_ids, start_positions, end_positions = \
            batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
        start_logits, end_logits = model(input_ids, segment_ids, input_mask)
        ignored_index = start_logits.size(1)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        duration = time.time() - start
        if batch_idx > 100:
            durations.append(duration)

        # print("bert one epoch end -----")

    print("Bert-base duration: {:.4f} s".format(np.mean(durations)))

if __name__ == "__main__":

    mp.set_start_method('spawn')
    barrier = Barrier(2)

    processes = []
    p1 = Process(target=bert_worker, args=(barrier, 75))
    p2 = Process(target=resnet50_worker, args=(barrier, 25))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    