##################################################
# Import libraries
##################################################
import  argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_index', type=int, help="Index of the GPU to use.",  default=2)
parser.add_argument('--seed',       type=int, help="Random seed.",              default=42)
parser.add_argument('--pretrained', type=str, help="The path of the pretrained encoder.",     default=None)
args = parser.parse_args()
cuda_index      = args.cuda_index
seed            = args.seed
path_encoder    = Path(args.pretrained) if args.pretrained is not None else None

import  sys
sys.stdout = open("log__deep_svdd.log", "w")
sys.stderr = sys.stdout

print(f"Importing libraries...", end=' ', flush=True)
from    pathlib             import  Path
import  torch
from    torch.utils.data    import  DataLoader
print(f"Done.", flush=True)

print(f"Importing custom modules...", end=' ', flush=True)
sys.path.append('../')
from    utils           import  *
from    models          import  *
from    dataloaders     import  *
print(f"Done.", flush=True)


device = torch.device(f'cuda:2')

# %%
BATCH_SIZE      = 1024
LR              = 1e-3
LR_AE           = 1e-3
LR_MILESTONES   = [50]
NUM_EPOCHS      = 75
NUM_EPOCHS_AE   = 150
WEIGHT_DECAY    = 5e-7
WEIGHT_DECAY_AE = 5e-3

NORMAL_CLASS    = [0]


print(f"Preparing data loaders...", end=' ', flush=True)
train_dataset   = Dataset_MNIST(BASE_PATHS[MNIST]/"train.npz", NORMAL_CLASS)
test_dataset    = Dataset_MNIST(BASE_PATHS[MNIST]/"test.npz")
train_loader    = DataLoader(train_dataset, BATCH_SIZE)
test_loader     = DataLoader(test_dataset, BATCH_SIZE)
print(f"Done.", flush=True)
model_ae        = Autoencoder_MNIST(device)
mse_loss_ae     = torch.nn.MSELoss(reduction="mean")
optimizer_ae    = torch.optim.Adam(model_ae.parameters(), LR_AE, weight_decay=WEIGHT_DECAY_AE)
scheduler_ae    = torch.optim.lr_scheduler.MultiStepLR(optimizer_ae, LR_MILESTONES, gamma=0.1)
list_loss_ae:   list[float] = []


if path_encoder is None:
    print(f"Initiate the training of the autoencoder", flush=True)
    model_ae.train()
    for epoch in range(1, 1+NUM_EPOCHS_AE):
        avg_loss_ae = 0
        for data, _ in train_loader:
            data = data.to(device)
            
            reconst = model_ae.forward(data)
            loss_ae = mse_loss_ae.forward(reconst, data)
            optimizer_ae.zero_grad()
            loss_ae.backward()
            optimizer_ae.step()
            
            avg_loss_ae += loss_ae.item()*data.size(0)
        avg_loss_ae /= len(train_dataset)
        list_loss_ae.append(avg_loss_ae)
        scheduler_ae.step()
        
        if epoch%5==0 or epoch==1:
            print(f"Epoch {epoch:03d} >>> Train loss {avg_loss_ae:.4e}", flush=True)
    list_loss_ae = torch.tensor(list_loss_ae)
    print(f"Finished the training of the autoencoder", flush=True)
elif Path(path_encoder).exists():
    print(f"Loading the pretrained encoder from {path_encoder}...", end=' ', flush=True)
    state_dict = torch.load(path_encoder, map_location=device)
    model_ae.load_state_dict(state_dict)
    print(f"Done.", flush=True)
else:
    raise FileNotFoundError(f"The pretrained encoder at [{str(path_encoder)}] does not exist.")


model       = DeepSVDD(model_ae.encoder, nu=5e-2, C=1e-3, is_soft_boundary=False, device=device)
optimizer   = torch.optim.Adam(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
scheduler   = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=0.1)
list_loss:  list[float] = []

print(f"Initiate the training of Deep SVDD.", flush=True)
model.initialize_center(dataloader=train_loader)
model.train()
for epoch in range(1, 1+NUM_EPOCHS):
    avg_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        loss = model.compute_loss__one_class(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()*data.size(0)
    avg_loss /= len(train_dataset)
    list_loss.append(avg_loss)
    scheduler.step()
    
    if epoch%5==0 or epoch==1:
        print(f"Epoch {epoch:03d} >>> Train loss {avg_loss:.4e}", flush=True)
list_loss = torch.tensor(list_loss)
print(f"Finished the training of Deep SVDD.", flush=True)


print(f"Evaluate the trained model.", flush=True)
model.eval()
prediction = []
ground_truth = []
model.eval()
with torch.inference_mode():
    for data, target in test_loader:
        is_normal = (target.unsqueeze(-1)==torch.tensor(NORMAL_CLASS, dtype=target.dtype)).any(dim=-1)
        _gt     = torch.where(is_normal, 0, 1).cpu()
        _pred   = model.anomaly_score(data.to(device)).cpu()
        ground_truth.append(_gt)
        prediction.append(_pred)
prediction   = torch.cat(prediction, dim=0)
ground_truth = torch.cat(ground_truth, dim=0)

auc_score = roc_auc_score(prediction, ground_truth)
print(f"AUROC score >>> {auc_score:.4f}", flush=True)


##################################################
##################################################
# End of file