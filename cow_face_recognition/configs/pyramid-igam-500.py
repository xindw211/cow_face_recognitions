import os

from easydict import EasyDict as edict

#
config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "igam_ir100"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.batch_size = 32
# 可重复性
config.reproducible = True

config.optimizer = "adam"
config.lr = 0.001
config.weight_decay = 0.001
config.gradient_acc = 2

config.rec = os.path.join(os.getcwd(), 'datasets/pyramid', 'train_cow')
config.num_classes = 522
config.num_image = 11134
config.num_epoch = 500
config.warmup_epoch = 75
config.val_targets = ['test_cow']

