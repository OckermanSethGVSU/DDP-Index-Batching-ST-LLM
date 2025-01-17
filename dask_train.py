import torch
import numpy as np
import argparse
import time
import util
import os
from util import *
import random
from model_ST_LLM import ST_LLM
from ranger21 import Ranger

# Torch dist
import torch.optim as optim
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# dask
from dask.distributed import LocalCluster
from dask.distributed import Client
from dask_pytorch_ddp import dispatch, results

# misc
import pandas as pd
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="bike_drop", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=64, help="number of features")
parser.add_argument("--num_nodes", type=int, default=250, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument("--epochs", type=int, default=500, help="") # 500
parser.add_argument("--print_every", type=int, default=1, help="")
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
    help="save path",
)
parser.add_argument(
    "--es_patience",
    type=int,
    default=100,
    help="quit if no improvement after this many iterations",
)
parser.add_argument(
    "--mode",
    type=str,
    default="local",
    help="local or distributed dask",
)

parser.add_argument(
    "--npar",
    type=int,
    default=1,
    help="number of parallel workers",
)

args = parser.parse_args()


class trainer:
    def __init__(
        self,
        scaler,
        input_dim,
        channels,
        num_nodes,
        input_len,
        output_len,
        dropout,
        lrate,
        wdecay,
        device,
    ):
        self.model = ST_LLM(
            device, input_dim, channels, num_nodes, input_len, output_len, dropout
        )
        
        # They do use the word token embedding layer (as they generate their own embeddings).
        # This creates issues when using DDP, thus freeze it
        # print()
        

        self.model.gpt.gpt2.wte.weight.requires_grad = False
        self.model = DDP(self.model, gradient_as_bucket_view=True).to(device)
        
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5
        print("The number of parameters: {}".format(self.model.module.param_num()))
        print(self.model)
        # exit()

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True  
    torch.manual_seed(seed)


def my_train(args, start_time):
    worker_rank = int(dist.get_rank())
    if args.mode == 'dist':
        device = f"cuda:{worker_rank % 4}"
        print(device)
        # print(train_dict)
        # device=None
        torch.cuda.set_device(worker_rank % 4)
    else:
        device = None
    
    pre_start = time.time()
    my_data, mean, std, x_train, x_val, x_test = index_load_dataset(args.data, "df")
    pre_end = time.time()
    
    
    train_dataset = IndexDataset(x_train,my_data)
    val_dataset = IndexDataset(x_val,my_data)

    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, sampler= train_sampler, batch_size=args.batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset,sampler= val_sampler, batch_size=args.batch_size, drop_last=True)


    scaler = StandardScaler(mean=mean, std=std)

    loss = 9999999
    test_log = 999999
    epochs_since_best_mae = 0
    path = args.save + "/"

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []
    
    # if not os.path.exists(path):
    #     os.makedirs(path)

    engine = trainer(
        scaler,
        args.input_dim,
        args.channels,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
    )

    print("start training...", flush=True)
    

    
    train_start = time.time()
    for i in range(1, args.epochs + 1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []
        epoch_start = time.time()
        t1 = time.time()
        # dataloader['train_loader'].shuffle()
        for i, (x, y) in enumerate(train_loader):
            
            trainx = x.to(device).float()
            trainx[...,0] = (trainx[...,0] -  mean) / std
            # print(trainx)
            
            trainx = trainx.transpose(1, 3)
            trainy = y.to(device).float()
            # trainy[..., 0] = scaler.inverse_transform(y[...,0])
           
            # print(trainy.shape)
            trainy = trainy.transpose(1, 3)
            
            
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])
            print(f"{i + 1}/{len(train_loader)}" , end="\r")
            print(metrics)


            
        
        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()
        for i, (x, y) in enumerate(val_loader):
            testx = torch.Tensor(x).to(device).float()
            testx[...,0] = (testx[...,0] -  mean) / std
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device).float()
            # testy[..., 0] = scaler.inverse_transform(testy[...,0])
            testy = testy.transpose(1, 3)
            
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])
            print(f"{i + 1}/{len(val_loader)}" , end="\r")
            print(metrics)


           

        s2 = time.time()

        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        
        metric_tensor = torch.tensor((mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape))
        dist.reduce(metric_tensor, 0)
        metric_tensor /= dist.get_world_size()
        

        train_m = dict(
            epoch_runtime=s2 - t1,
            train_runtime=t2 - t1,
            val_runtime=s2 - s1,
            train_loss=np.mean(train_loss),
            train_rmse=np.mean(train_rmse),
            train_mape=np.mean(train_mape),
            train_wmape=np.mean(train_wmape),
            valid_loss=metric_tensor[0],
            valid_rmse=metric_tensor[1],
            valid_mape=metric_tensor[2],
            valid_wmape=metric_tensor[3],
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, "
        print(
            log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape),
            flush=True,
        )


        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"train.csv")

    end_time = time.time()

    #     )
    with open("stats.txt", "a") as file:
                    file.write(f"total_time: {end_time - start_time}\n")
                    file.write(f"preprocess_time: {pre_end - pre_start}\n")
                    file.write(f"training_time: {end_time - train_start}\n")

                    file.write(f"train_opt_loss: {train_csv['train_loss'].min()}\n")
                    file.write(f"train_opt_rmse: {train_csv['train_rmse'].min()}\n")
                    file.write(f"train_opt_mape: {train_csv['train_mape'].min()}\n")
                    file.write(f"train_opt_wmape: {train_csv['train_wmape'].min()}\n")
                    
                    file.write(f"valid_opt_loss: {train_csv['valid_loss'].min()}\n")
                    file.write(f"valid_opt_rmse: {train_csv['valid_rmse'].min()}\n")
                    file.write(f"valid_opt_mape: {train_csv['valid_mape'].min()}\n")
                    file.write(f"valid_opt_wmape: {train_csv['valid_wmape'].min()}\n")




def main():
    seed_it(6666)
    start_time = time.time()
    data = args.data

    if args.data == "bike_drop":
        args.data = "data//" + args.data
        args.num_nodes = 250
    
    elif args.data == "bike_pick":
        args.data = "data//" + args.data
        args.num_nodes = 250
    
    elif args.data == "taxi_drop":
        args.data = "data//" + args.data
        args.num_nodes = 266

    elif args.data == "taxi_pick":
        args.data = "data//" + args.data
        args.num_nodes = 266
    elif args.data == 'metr-la':
        args.data = "data/" + args.data + f"/{args.data}.h5"
        args.num_nodes = 207
    elif args.data == 'pems-bay':
        args.data = "data/" + args.data + f"/{args.data}.h5"
        args.num_nodes = 325
    
    if args.mode == 'local':
            cluster = LocalCluster(n_workers=args.npar)
            client = Client(cluster)
    elif args.mode == 'dist':
        client = Client(scheduler_file = f"cluster.info")
    else:
        print(f"{args.mode} is not a valid mode; Please enter mode as either 'local' or 'dist'")
        exit()

    if args.mode == "dist":
            for f in ['util.py', 'model_ST_LLM.py', 'ranger21.py', 'dask_train.py']:
                print("Uploading ", f, flush=True)
                client.upload_file(f)
   
    futures = dispatch.run(client, my_train,
                                args=args,
                                start_time=start_time,
                                backend="gloo")
    key = uuid.uuid4().hex
    rh = results.DaskResultsHandler(key)
    rh.process_results(".", futures, raise_errors=False)
    client.shutdown()
    


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
