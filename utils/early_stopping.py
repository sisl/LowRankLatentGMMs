class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        Args:
            patience (int): Number of epochs to wait after the last improvement.
            delta (float): Minimum change in validation loss to qualify as an improvement.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            print(f"Validation loss improved to {val_loss:.4f}.")
            return True
        else:
            self.counter += 1
            print(f"No improvement in validation loss for {self.counter} epoch(s).")

            if self.counter >= self.patience:
                self.should_stop = True
                print("Early stopping triggered.")
            return False