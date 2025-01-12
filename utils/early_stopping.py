class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait after the last improvement.
            delta (float): Minimum change in validation loss to qualify as an improvement.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0
        self.early_stop = False

    def __call__(self, val_loss, logger):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            if self.verbose:
                logger.info(f"Validation loss improved to {val_loss:.4f}.")
        else:
            self.epochs_without_improvement += 1
            if self.verbose:
                logger.info(f"No improvement in validation loss for {self.epochs_without_improvement} epoch(s).")

            if self.epochs_without_improvement >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered.")

