import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix A Notes - PyTorch Introduction

    ## Tensor Creation and DataTypes
    """)
    return


@app.cell
def _():
    import torch
    tensor1d = torch.tensor([1, 2, 3])
    _tensor2d = torch.tensor([[1, 2], [3, 4]])
    tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    return tensor1d, torch


@app.cell
def _(tensor1d, torch):
    print("tensor1d.dtype = " + str(tensor1d.dtype))

    floatvec = torch.tensor([1.0, 2.0, 3.0])
    print("floatvec.dtype = " + str(floatvec.dtype))

    # Can use .to method of a tensor to change dtype
    floatvec = tensor1d.to(torch.float32)
    print("floatvec.dtype = " + str(floatvec.dtype))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Common Tensor Operations
    """)
    return


@app.cell
def _(torch):
    _tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print('tensor2d.shape = ' + str(_tensor2d.shape))
    print(_tensor2d.reshape(3, 2))
    print(_tensor2d.view(3, 2))
    print(_tensor2d.T)
    # Reshaping
    print(_tensor2d.matmul(_tensor2d.T))  # Added later for NumPy agreement (& more robust)
    # Transpose
    # Matrix multiplication
    print(_tensor2d @ _tensor2d.T)  # Original Lua compliance
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Computation Graphs
    """)
    return


@app.cell
def _(torch):
    import torch.nn.functional as F  # Common convention in PyTorch
    _y = torch.tensor([1.0])
    _x1 = torch.tensor([1.1])  # target label
    w1 = torch.tensor([2.2])  # input feature
    b = torch.tensor([0.0])  # weight parameter
    _z = _x1 * w1 + b  # bias unit
    _a = torch.sigmoid(_z)
    _loss = F.binary_cross_entropy(_a, _y)
    return (F,)


@app.cell
def _(mo):
    mo.image(src="imgs/A_computation_graph.png")
    return


@app.cell
def _(mo):
    mo.image(src="imgs/A_gradient_graph.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyTorch builds a computational graph internally if one of its terminal nodes has the ```requires_grad``` attribute as ```True```. Gradients are required for training neural networks by the popular backpropagation algorithm.

    Calling the ```grad``` function computes the gradient of the loss concerning a model parameter (below).
    """)
    return


@app.cell
def _(F, torch):
    from torch.autograd import grad
    _y = torch.tensor([1.0])
    _x1 = torch.tensor([1.1])
    w1_1 = torch.tensor([2.2], requires_grad=True)
    b_1 = torch.tensor([0.0], requires_grad=True)
    _z = _x1 * w1_1 + b_1
    _a = torch.sigmoid(_z)
    loss_1 = F.binary_cross_entropy(_a, _y)
    grad_L_w1 = grad(loss_1, w1_1, retain_graph=True)
    grad_L_b = grad(loss_1, b_1, retain_graph=True)
    print(grad_L_w1)
    print(grad_L_b)
    return b_1, loss_1, w1_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This was manual grad computation. PyTorch provides higher-level access to grads through calling ```.backward()``` on the loss. This asks PyTorch to compute the gradients of all the leaf nodes in the graph, which are stored in the tensor's ```.grad``` attributes.
    """)
    return


@app.cell
def _(b_1, loss_1, w1_1):
    loss_1.backward()
    print(w1_1.grad)
    print(b_1.grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementing Multilayer Neural Networks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A **multilayer perceptron** is a fully connected neural network with nonlinear activation layers. Uses a continuous nonlinear function, like the sigmoid or softmax to allow for the calculation of gradients for backpropagation.

    We subclass ```torch.nn.Module``` to define our own neural network. This gives us a lot of ready-to-go functionality and customizability. Layers are defined in the ```__init__``` constructor. Interactions between layers are defined in the ```forward``` method.

    We do not usually need to implement our own ```backward``` method, since it is used for training for gradients of loss.

    The following is a multilayer perceptron with two hidden layers:
    """)
    return


@app.cell
def _(torch):
    class NeuralNetwork(torch.nn.Module):

        def __init__(self, num_inputs, num_outputs):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(num_inputs, 30), torch.nn.ReLU(), torch.nn.Linear(30, 20), torch.nn.ReLU(), torch.nn.Linear(20, num_outputs))

        def forward(self, x):  # Sequential isn't required, but it makes the forward method simpler
            logits = self.layers(x)
            return logits  # 1st hidden layer  # nonlinear activation functions are placed between layers  # 2nd hidden layer  # previous outputs = current inputs  # output layer  # name for outputs of the last layer

    return (NeuralNetwork,)


@app.cell
def _(NeuralNetwork):
    _model = NeuralNetwork(50, 3)

    print(_model)
    return


@app.cell
def _(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters:", num_params)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A ```Linear``` layer multiplies the inputs with a weight matrix and adds a bias vector. Also referred to as a *feedforward* or *fully connected* layer.
    """)
    return


@app.cell
def _(model):
    # Access the weight parameter matrix at position 0
    print(model.layers[0].weight)
    print(model.layers[0].weight.shape)

    # Similar for bias vector
    print(model.layers[0].bias)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note ```requires_grad=True``` is present for the weight and bias tensors. This is the default behavior of the ```Linear``` layer.

    Initial model weights with small random numbers is ideal for breaking symmetry during training for deep learning. We can make the initialized weights reproducible with ```manual_seed```.
    """)
    return


@app.cell
def _(NeuralNetwork, torch):
    torch.manual_seed(123)
    model_1 = NeuralNetwork(50, 3)
    print(model_1.layers[0].weight)
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NeuralNetwork used in a forward pass
    """)
    return


@app.cell
def _(model_1, torch):
    torch.manual_seed(123)
    X = torch.rand((1, 50))
    _out = model_1(X)
    print(_out)
    with torch.no_grad():
        _out = model_1(X)
    print(_out)
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyTorch uses the ```grad_fn``` information to retrace the functions in its backward pass.

    To compute class-membership probabilities, we have to call ```softmax``` explicitly. PyTorch combines softmax with the negative log-likelihood loss for numerical efficiency and stability by default.
    """)
    return


@app.cell
def _(X, model_1, torch):
    with torch.no_grad():
        _out = torch.softmax(model_1(X), dim=1)
    print(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up efficient data loaders
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyTorch has a ```Dataset``` for defining how data records are loaded. PyTorch has a ```DataLoader``` class for handling data shuffling and batch assembly.

    Here's a small toy dataset with five training examples with two features each. There is also a tensor with the class labels for the examples.
    """)
    return


@app.cell
def _(torch):
    x_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    # PyTorch requires class labels start with 0
    # Largest label can be no more than (# logits) - 1
    y_train = torch.tensor([0, 0, 0, 1, 1])

    x_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6]
    ])

    y_test = torch.tensor([0, 1])
    return x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Custom Dataset implementation:
    """)
    return


@app.cell
def _():
    from torch.utils.data import Dataset

    class ToyDataset(Dataset):
        # used for defining the references to the training data and labels
        def __init__(self, x, y):
            self.features = x
            self.labels = y

        # retrieves one data record and it's corresponding label
        def __getitem__(self, index):
            one_x = self.features[index]
            one_y = self.labels[index]
            return one_x, one_y 

        def __len__(self):
            return self.labels.shape[0]

    return (ToyDataset,)


@app.cell
def _(ToyDataset, x_test, x_train, y_test, y_train):
    train_ds = ToyDataset(x_train, y_train)
    test_ds = ToyDataset(x_test, y_test)
    return test_ds, train_ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Custom DataLoader implementation:
    """)
    return


@app.cell
def _(test_ds, torch, train_ds):
    from torch.utils.data import DataLoader

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0, # number of background processes
        drop_last=True # to prevent uneven batch sizes
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    return test_loader, train_loader


@app.cell
def _(train_loader):
    for idx, (x,y) in enumerate(train_loader):
        print(f"Batch {idx+1}:", x, y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Typical Training Loop
    """)
    return


@app.cell
def _(F, NeuralNetwork, torch, train_loader):
    torch.manual_seed(123)

    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.5
    )

    num_epochs = 3
    for epoch in range(num_epochs):

        model.train()
    
        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad() # prevent gradient accumulation
            loss.backward()
            optimizer.step() # optimizer updates model parameters with gradients

            ### LOGGING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                  f" | Train Loss: {loss:.3f}")

    model.eval()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```lr``` (learning rate) and ```num_epochs``` are *hyperparameters* - tunable settings that we must experiment with based on experimental observations.

    This implementation uses a stochastic gradient descent (```SGD```) optimizer to adjust the parameters by multiplying the gradients with the learning rate and adding the scaled negative gradient to the parameters.

    With a trained model, we can now make predictions:
    """)
    return


@app.cell
def _(model, torch, x_train):
    model.eval()
    with torch.no_grad():
        outputs = model(x_train)
    print(outputs)

    # Use softmax to obtain class membership probabilities
    torch.set_printoptions(sci_mode=False)
    probas = torch.softmax(outputs, dim=1)
    print(probas)

    # Convert into class label predictions
    predictions = torch.argmax(probas, dim=1)
    print(predictions)

    # Skip intermediate step of computing softmax since these classes are defined by columns with max values
    predictions = torch.argmax(outputs, dim=1)
    print(predictions)
    return (predictions,)


@app.cell
def _(predictions, torch, y_train):
    torch.sum(predictions == y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's generalize the computation of the prediction accuracy:
    """)
    return


@app.cell
def _(torch):
    def compute_accuracy(model, dataloader):

        model.eval()
        correct = 0.0
        total_examples = 0

        for idx, (features, labels) in enumerate(dataloader):

            with torch.no_grad():
                logits = model(features)

            predictions = torch.argmax(logits, dim=1)
            compare = labels == predictions
            correct += torch.sum(compare)
            total_examples += len(compare)

        return (correct / total_examples).item()

    return (compute_accuracy,)


@app.cell
def _(compute_accuracy, model, test_loader, train_loader):
    print(compute_accuracy(model, train_loader))
    print(compute_accuracy(model, test_loader))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    => The model is trained and accurate for the provided test data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving and Loading Models
    """)
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(), "model.pth") # .pth and .pt are conventional
    return


@app.cell
def _(NeuralNetwork, torch):
    newModel = NeuralNetwork(2,2) # architecture needs to match original saved model exactly
    newModel.load_state_dict(torch.load("model.pth"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimizing training with GPUs

    Cost of memory transfer from CPU to GPU means that this small dataset won't see much of a speedup, but larger deep neural networks will benefit much more.
    """)
    return


@app.cell
def _(F, NeuralNetwork, torch, train_loader):
    def _():
        torch.manual_seed(123)

        model = NeuralNetwork(num_inputs=2, num_outputs=2)

        ### NEW LINE FOR GPU (Apple Silicon)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        model.to(device)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.5
        )

        num_epochs = 3
        for epoch in range(num_epochs):

            model.train()
        
            for batch_idx, (features, labels) in enumerate(train_loader):
                ### NEW LINE FOR GPU
                features, labels = features.to(device), labels.to(device)
            
                logits = model(features)

                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad() # prevent gradient accumulation
                loss.backward()
                optimizer.step() # optimizer updates model parameters with gradients

                ### LOGGING
                print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                      f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                      f" | Train Loss: {loss:.3f}")
    
        return model.eval()

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
