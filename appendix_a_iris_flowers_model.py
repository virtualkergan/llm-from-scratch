import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    from torch.autograd import grad
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader

    return Dataset, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # *Iris* flowers classification with neural network

    This model is trained on the *Iris* flower data set used in Ronald Fisher's 1936 paper, *The use of multiple measurements in taxonomic problems*. The concepts used here were learned from Appendix A of Sebastian Raschka's book, *Build a Large Language Model (From Scratch)*.

    The data set has instances that belong to one of three classifications of *Iris* flowers:

    1. *l. setosa*
    2. *l. versicolor*
    3. *l. virginica*

    For each instance, four measurements are recorded:

    1. Sepal length
    2. Sepal width
    3. Petal length
    4. Petal width
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function Definitions
    """)
    return


@app.cell
def _(torch):
    class NeuralNetwork(torch.nn.Module):
        def __init__(self, num_inputs, num_outputs):
            super().__init__()
        
            self.layers = torch.nn.Sequential(

                # 1st hidden layer
                torch.nn.Sequential(torch.nn.Linear(num_inputs, 30),
                torch.nn.ReLU(),

                # 2nd hidden layer
                torch.nn.Linear(30, 20), torch.nn.ReLU(),
                torch.nn.Linear(20, num_outputs))
            )
    
        def forward(self, x):  
            logits = self.layers(x)
            return logits

    return


@app.cell
def _(Dataset):
    class IrisDataset(Dataset):
        def __init__(self, x, y):
            self.measurements = x
            self.labels = y

        def __getitem__(self, index):
            one_x = self.measurements[index]
            one_y = self.labels[index]
            return one_x, one_y

        def __len__(self):
            return self.labels.shape[0]

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Initialization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Loop
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Testing
    """)
    return


if __name__ == "__main__":
    app.run()
