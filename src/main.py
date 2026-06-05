import torch
from model.neural_network import Model

def main():
    # Instantiate and load model with the weights
    model = Model()
    model.load_state_dict(torch.load("../mnist_model.pt"))
    model.eval()

    # Build UI


if __name__ == "__main__":
    main()
