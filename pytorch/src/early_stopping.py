import numpy as np


class EarlyStopping:
    def __init__(self, early_stopping_patience: int, logger, **kwargs):
        assert early_stopping_patience > 0
        self.count = 0
        self.best_loss = np.inf
        self.patience = early_stopping_patience
        self.logger = logger
        self.logger.info(f"Early stopping patience = {self.patience}")

    def __call__(self, current_loss: float) -> bool:

        if current_loss > self.best_loss:
            self.count += 1

            if self.count >= self.patience:
                self.logger.info(
                    f"Early stopped: count = {self.count} (>= {self.patience})"
                )
                return True

            self.logger.info(f"Early stopping count = {self.count}")

        else:
            self.count = 0
            self.best_loss = current_loss
            self.logger.info("Early stopping count is reset.")

        return False