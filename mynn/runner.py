import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, eval_iters=None, augment_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.eval_iters = eval_iters
        self.augment_fn = augment_fn

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.model.set_training(True)
        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            dev_score, dev_loss = None, None
            last_eval_iter = -1
            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                if self.augment_fn is not None:
                    train_X = self.augment_fn(train_X)

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)

                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # Only evaluate at log intervals to avoid excessive overhead
                if iteration % log_iters == 0:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    last_eval_iter = iteration
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

                # Record last known dev metrics for plotting continuity
                if dev_score is not None:
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                elif self.dev_scores:
                    self.dev_scores.append(self.dev_scores[-1])
                    self.dev_loss.append(self.dev_loss[-1])
                else:
                    self.dev_scores.append(0)
                    self.dev_loss.append(0)

            # Evaluate at end of epoch if not already done at last iteration
            if dev_score is None or last_eval_iter < int(X.shape[0] / self.batch_size):
                dev_score, dev_loss = self.evaluate(dev_set)
                print(f"epoch: {epoch}, final dev loss: {dev_loss:.4f}, score: {dev_score:.4f}")

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score

    def evaluate(self, data_set):
        self.model.set_training(False)
        X, y = data_set
        logits = self.model(X)
        temp_loss_fn = type(self.loss_fn)(model=self.model, max_classes=self.loss_fn.max_classes)
        loss = temp_loss_fn(logits, y)
        score = self.metric(logits, y)
        self.model.set_training(True)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)