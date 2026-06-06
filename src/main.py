import torch
from model.neural_network import Model
from gui.app import App

def main():
     # Create model
    model = Model()

    # Load learned weights
    model.load_state_dict(torch.load("./mnist_model.pt"))
    
    # Switch to inference mode
    model.eval()

    # Start GUI
    App(model)

    
if __name__ == "__main__":
    main()
