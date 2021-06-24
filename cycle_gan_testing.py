import torch
import torch.nn as nn

from torch.utils import data
from torchvision.utils import save_image

from data_generator import CombinedDataset, ArtificialDataset
from resunet import ResUnet
from discriminator import Discriminator


# Setting parameters and stuff
#val_path = "data/val/grid/"
nogrid_path = "data/val/nogrid/"
grid_path = "data/val/grid/"

use_cuda = torch.cuda.is_available()
print("Using GPU:", use_cuda)
torch.backends.cudnn.deterministic=True
processor = torch.device("cuda:0" if use_cuda else "cpu")
max_imgs = 100000
batch_size=1
augments = {"crop" : (400,400),
            "hflip" : 0.5,
            "vflip" : 0.5,
            "angle" : 0,
            "shear" : 0,
            "brightness" : (0.75,1.25),
            "pad" : (0,0,0,0),
            "contrast" : (0.5,1.25)}

artificial_augments = {"grid_size" : (30,90),
                       "grid_intensity" : (-1,1),
                       "grid_offset_x" : (0,90),
                       "grid_offset_y" : (0,90),
                       "crop" : (400,400)}
# Loading models
loaded_params = torch.load("final_grid_gen.pth")
grid_gen = ResUnet(**loaded_params["args_dict"]).to(processor)
grid_gen.load_state_dict(loaded_params["state_dict"])

loaded_params = torch.load("final_nogrid_gen.pth")
nogrid_gen = ResUnet(**loaded_params["args_dict"]).to(processor)
nogrid_gen.load_state_dict(loaded_params["state_dict"])

loaded_params = torch.load("final_grid_disc.pth")
grid_disc = Discriminator(**loaded_params["args_dict"]).to(processor)
grid_disc.load_state_dict(loaded_params["state_dict"])

loaded_params = torch.load("final_nogrid_disc.pth")
nogrid_disc = Discriminator(**loaded_params["args_dict"]).to(processor)
nogrid_disc.load_state_dict(loaded_params["state_dict"])

loss_function = nn.MSELoss().to(processor)

# Loading dataset
validation_set = ArtificialDataset (max_imgs, nogrid_path, **artificial_augments, seed=1337)
#validation_set = CombinedDataset (max_imgs, grid_path, nogrid_path, **augments, seed=1337)
validation_generator = data.DataLoader (validation_set, batch_size=batch_size, shuffle=False, num_workers=8)


val_grid_loss = 0
val_nogrid_loss = 0
nogrid_disc.eval()
nogrid_gen.eval()
grid_disc.eval()
grid_gen.eval()

#for i, (batch, labels) in enumerate(validation_generator):
    #batch = batch.to(processor)
    #labels = labels.to(processor)
for i, batch in enumerate(validation_generator):
    real_grid = batch["grid"].to(processor)
    real_nogrid = batch["nogrid"].to(processor)

    fake_label = torch.full((batch_size, 1), 0, device=processor, dtype=torch.float32)

    fake_grid = grid_gen (real_nogrid)
    fake_nogrid = grid_gen (real_grid)

    fake_grid_pred = grid_disc (fake_grid)
    fake_nogrid_pred = nogrid_disc (fake_nogrid)

    loss_1 = loss_function(fake_grid_pred, fake_label)
    loss_2 = loss_function(fake_nogrid_pred, fake_label)

    val_grid_loss += loss_1.item()
    val_nogrid_loss += loss_2.item()

    # get this to output both images and the true image in comparison
    if real_grid.size()[0] == 1:
        #to_save = out
        to_save = torch.cat ((real_grid, fake_nogrid, real_nogrid, fake_grid), dim=2)
        #to_save = torch.cat ((labels, out), dim=2)
    else:
        #to_save = out.squeeze(0)
        to_save = torch.cat ((real_grid, fake_nogrid.squeeze(0), real_nogrid, fake_grid.squeeze(0)), dim=2)
        #to_save = torch.cat ((labels, out.squeeze(0), batch), dim=2)
        #to_save = torch.cat ((labels, out.squeeze(0)), dim=2)
        #print(to_save.size())
    save_image(to_save, "data/artificial_val/{0}.png".format(i))

val_grid_loss = val_grid_loss/len(validation_generator)
print ("Average grid loss over validation data:", val_grid_loss)

val_nogrid_loss = val_nogrid_loss/len(validation_generator)
print ("Average nogrid loss over validation data:", val_nogrid_loss)
