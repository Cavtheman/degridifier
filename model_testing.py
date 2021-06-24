import torch
import torch.nn as nn

from torch.utils import data
from torchvision.utils import save_image

from data_generator import ArtificialDataset
from resunet import ResUnet
from unet import Unet


# Setting parameters and stuff
#val_path = "data/val/grid/"
val_path = "data/test/nogrid/"
use_cuda = torch.cuda.is_available()
print("Using GPU:", use_cuda)
torch.backends.cudnn.deterministic=True
processor = torch.device("cuda:0" if use_cuda else "cpu")
max_imgs = 100000
batch_size=2
augments = {"grid_size" : (30,90),
            "grid_intensity" : (-1,1),
            "grid_offset_x" : (0,90),
            "grid_offset_y" : (0,90),
            "crop" : (400,400),
            "hflip" : 0.5,
            "vflip" : 0.5,
            "angle" : 0,
            "shear" : 0,
            "brightness" : (0.75,1.25),
            "pad" : (0,0,0,0),
            "contrast" : (0.5,1.25)}

# Loading model
#loaded_params = torch.load("saved/unet/tanh_clamp_early.pth")
loaded_params = torch.load("temp_model.pth")
model = ResUnet(**loaded_params["args_dict"]).to(processor)
model = Unet(**loaded_params["args_dict"]).to(processor)
model.load_state_dict(loaded_params["state_dict"])

loss_function = nn.L1Loss().to(processor)

# Loading dataset
validation_set = ArtificialDataset(max_imgs, val_path, **augments, seed=1337)
validation_generator = data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8)


val_loss = 0
model.eval()
for i, data in enumerate(validation_generator):
    batch = data["grid"].to(processor)
    labels = data["nogrid"].to(processor)

    out = model(batch)
    loss = loss_function(out, labels)
    val_loss += loss.item()

    if batch.size()[0] == 1:
        #to_save = out
        to_save = torch.cat ((labels, out, batch), dim=2)
        #to_save = torch.cat ((labels, out), dim=2)
    else:
        #to_save = out.squeeze(0)
        to_save = torch.cat ((labels, out.squeeze(0), batch), dim=2)
        #to_save = torch.cat ((labels, out.squeeze(0)), dim=2)
        #print(to_save.size())
    save_image(to_save, "data/artificial_val/{0}.png".format(i))

val_loss = val_loss/len(validation_generator)

print ("Average L1 loss over test data:", val_loss)
