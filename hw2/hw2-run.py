import os
import argparse
from scipy.io import loadmat
import matplotlib.pyplot as plt
from nn import *
from hw2 import *

parser = argparse.ArgumentParser(description="Customize the parameters for neural network model.")

# Add arguments
parser.add_argument("-f", "--file", help="Path to image file", type=str)
parser.add_argument("-i", "--img", help="Image index", type=int, default=1)
parser.add_argument("-o", "--output", help="Output file path", type=str, default="./")
parser.add_argument("-s", "--structure", help="Linear layer structure, with ',' as delimiter", type=str, default="512,512")

parser.add_argument("--optim", help="Optimizer type, 'SGD' or 'Adam'", type=str, default="SGD")
parser.add_argument("-b", "--batch_size", help="Training batch size", type=int, default=128)
parser.add_argument("-e", "--epoch", help="Training epoch count", type=int, default=500)
parser.add_argument("-l", "--lr", help="Learning rate", type=float, default=1E-3)
parser.add_argument("-v", "--verbose", help="verbose output", type=bool, default=False)

# Parse arguments
args = parser.parse_args()

# Parameters parsing
img_file = args.file  # image file path
output_dir = args.output
output_dir = output_dir if output_dir[-1] == "/" else output_dir + "/"# output directory
idx = args.img
structure = args.structure
neuron = [int(i.strip()) for i in structure.split(',')]
optimizer_type = args.optim
batch_size = args.batch_size
epoch = args.epoch
lr = args.lr
verbose = args.verbose
print(neuron)

# Check existence of output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data
content = loadmat(img_file)
X1 = np.array(content['X1'])
Y1 = np.array(content['Y1'])
X2 = np.array(content['X2'])
Y2 = np.array(content['Y2'])
# print(X1.shape, Y1.shape, X2.shape, Y2.shape)
# img1 = convert_img(X1, Y1)
# img2 = convert_img(X2, Y2)
# plt.imshow(img1, cmap='gray')
# plt.show()
# plt.imshow(img2)
# plt.show()
print("Data loaded.")

if idx == 1:
    X, Y = X1, Y1
elif idx == 2:
    X, Y = X2, Y2
else:
    raise ValueError(f"Unknown image index: {idx}")

# Layer structure analysis
input_dim = X.shape[1]
output_dim = Y.shape[1]
dims = [input_dim, *neuron, output_dim]
layers = []
flag = True
for i in range(len(dims)-1):
    if flag:
        layers.append(Linear(dims[i], dims[i+1], is_input=True))
        flag = False
    else:
        layers.append(Linear(dims[i], dims[i+1]))
    layers.append(Sigmoid())
layers = layers[:-1]

model = Network(*layers)  # model definition
print(f"Model defined: {layers}")

dataloader = DataLoader(X, Y, batch_size=batch_size, shuffle=True)  # data loader
if optimizer_type == "Adam":
    optimizer = Adam(model.parameters(), lr=lr)
elif optimizer_type == "SGD":
    optimizer = SGD(model.parameters(), lr=lr)
else:
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")
loss_fn = MeanErrorLoss()   # loss function

print("Start training...")
record = train(model, dataloader, optimizer, loss_fn, num_iterations=epoch, verbose=verbose)
print("Training completed.")

# Visualize the loss
plt.plot(np.arange(len(record)), record)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(output_dir + f"loss_{idx}_{optimizer_type}_{batch_size}_{epoch}_{lr}_{structure}.png")


# Visualize the mapping image
raw_img = convert_img(X, Y)
mapping_img = convert_img(X, model(X))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(raw_img, cmap='gray')
axs[0].axis('off')
# axs[0].set_title("Raw Picture")

axs[1].imshow(mapping_img, cmap='gray')
axs[1].axis('off')
# axs[1].set_title("Mapping Picture")

plt.savefig(output_dir + f"result_{idx}_{optimizer_type}_{batch_size}_{epoch}_{lr}_{structure}.png")
print("Result saved.")
print("All done.")