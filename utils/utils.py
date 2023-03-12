from gensim.models.callbacks import CallbackAny2Vec

class LineSentences(object):
    def __init__(self, filenames):
        self.filenames = filenames
    
    # memory-friendly iterator
    def __iter__(self):
        for filename in self.filenames:
            for line in open(filename, "r", encoding="utf-8"):
                yield line.strip().split()

# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss