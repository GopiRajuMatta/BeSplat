# Load data from .pth file
import torch
data = torch.load("output/camera_0_no_event_loss/chkpnt7000.pth")

# Check the type of the data
#print(f"Type of data: {type(data)}")

# If it's a tuple, inspect its contents
if isinstance(data, tuple):
    print(f"Length of tuple: {len(data)}")
    for i, item in enumerate(data):
        print(f"Item {i}: Type={type(item)}, Value={item}")

