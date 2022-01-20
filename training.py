import torch
import matplotlib.pyplot as plt
import seaborn as sns

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, EpochWise, Accuracy, ConfusionMatrix
from ignite.metrics import MetricsLambda
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler, global_step_from_engine
from ignite.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR, OneCycleLR

# Класс для тренеровки

class Training:
    def __init__(self, device, model, name, log_dir):
        self.device = device
        self.model = model
        self.name = name
        self.num_epochs = 0
        self.optimizer = None
        self.scheduler = None
        self.clip = None
        self.patience = None
        self.iter = 0

        # движок для трейна
        self.trainer = Engine(self.train_fn)
        # движок для валидации
        self.val_evaluator = Engine(self.valid_fn)
        # движок для теста
        self.test_evaluator = Engine(self.valid_fn)
        # прогресс бары
        self.train_pbar = ProgressBar(persist=True, bar_format="")
        self.val_pbar = ProgressBar(persist=True, bar_format="")
        self.test_pbar = ProgressBar(persist=True, bar_format="")
        # для вывода прогресс бара
        self.train_pbar.attach(self.trainer)
        self.val_pbar.attach(self.val_evaluator)
        self.test_pbar.attach(self.test_evaluator)
        # логер
        self.tb_logger = TensorboardLogger(log_dir)

    def load_config(self, opt, loss, num_epochs,
                    train_loader, valid_loader,
                    test_loader, class_names,
                    clip=None, patience=None):
        self.optimizer = opt
        self.loss = loss
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.class_names = class_names
        if clip is not None:
            self.clip = clip
        if patience is not None:
            self.patience = patience
        self.create_metrics()
        if self.name[: 3] == 'Vit':
            self.scheduler = LRScheduler(StepLR(optimizer=self.optimizer,
                                                step_size=1,
                                                gamma=0.7))
        if self.name[: 3] == 'VIT':
            self.scheduler = LRScheduler(OneCycleLR(self.optimizer,
                                                    max_lr=0.1,
                                                    pct_start=0.2,
                                                    total_steps=len(self.train_loader) * self.num_epochs,
                                                    div_factor=25,
                                                    final_div_factor=10))
        self.add_events()

    def create_metrics(self):
        # создаём метрики
        self.train_metrics = {'train loss': RunningAverage(output_transform=lambda x: x,
                                                           alpha=0.9)
                              }
        self.valid_metrics = {'valid loss': RunningAverage(output_transform=lambda x: x[2],
                                                           alpha=0.9),
                              'valid accuracy': Accuracy(output_transform=lambda x: [x[0], x[1]]),
                              }
        self.test_metrics = {'test accuracy': Accuracy(output_transform=lambda x: [x[0], x[1]]),
                             'test cm': ConfusionMatrix(num_classes=len(self.class_names))
                             }
        for i, class_name in enumerate(self.class_names):
            class_name_r = class_name + ' recall'
            class_name_p = class_name + ' precision'
            self.test_metrics[class_name_r] = MetricsLambda(self.recall_, self.test_metrics['test cm'], i)
            self.test_metrics[class_name_p] = MetricsLambda(self.precision_, self.test_metrics['test cm'], i)
            # прикрепляем их к соответствующим движкам
        for name, metric in self.train_metrics.items():
            metric.attach(self.trainer, name)
        for name, metric in self.valid_metrics.items():
            metric.attach(self.val_evaluator, name)
        for name, metric in self.test_metrics.items():
            metric.attach(self.test_evaluator, name)

        # для логера

        self.tb_logger.attach(self.trainer,
                              OutputHandler(tag=self.name + " training",
                                            metric_names=[key for key in self.train_metrics.keys()]),
                              event_name=Events.EPOCH_COMPLETED)
        self.tb_logger.attach(self.val_evaluator,
                              OutputHandler(tag=self.name + " validation",
                                            metric_names=[key for key in self.valid_metrics.keys()],
                                            global_step_transform=global_step_from_engine(self.trainer)),
                              event_name=Events.EPOCH_COMPLETED)
        self.tb_logger.attach(self.trainer,
                              OutputHandler(tag=self.name + " training",
                                            output_transform=lambda x: x),
                              event_name=Events.ITERATION_COMPLETED)
        self.tb_logger.attach(self.val_evaluator,
                              OutputHandler(tag=self.name + " validation",
                                            output_transform=lambda x: x[2],
                                            global_step_transform=self.global_step_transform),
                              event_name=Events.ITERATION_COMPLETED)

    def recall_(self, cm, i):
        cm = cm.float().numpy()
        return cm[i][i] / sum(cm[:, i]) if cm[i][i] != 0 else 0.

    def precision_(self, cm, i):
        cm = cm.float().numpy()
        return cm[i][i] / sum(cm[i, :]) if cm[i][i] != 0 else 0.

    def add_events(self):
        # добавляем вывод строки метрик после каждой эпохи тренировки
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_train_val)
        # добавляем запуск валидации
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.run_valid)
        self.val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.log_train_val)
        if self.scheduler is not None:
            if self.name[: 3] == 'Vit':
                self.val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.scheduler)
            elif self.name[: 3] == 'VIT':
                self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.scheduler)
        # добавляем запуск тестирования
        self.trainer.add_event_handler(Events.COMPLETED, self.run_test)
        self.test_evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.log_test)
        # добавим сохранение модели при улучшении
        checkpointer = ModelCheckpoint(dirname='/content/saved_models',
                                       filename_prefix='',
                                       score_function=self.score_function,
                                       score_name='valid accuracy',
                                       n_saved=2,
                                       create_dir=True,
                                       require_empty=False
                                       )
        self.val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer,{self.name: self.model})
        # добавляем остановку процесса обучения если лосс на валидации не уменьшается 2 эпох
        if self.patience is not None:
            stopping = EarlyStopping(patience=self.patience, score_function=self.score_function, trainer=self.trainer)
            self.val_evaluator.add_event_handler(Events.COMPLETED, stopping)
    # #     # добавляем запуск вычисления метрик на тестовой выборке после окончания обучения
    # #     # self.trainer.add_event_handler(Events.COMPLETED, self.run_test)

    def run_training_loop(self):
        self.trainer.run(self.train_loader, max_epochs=self.num_epochs)

    def run_valid(self):
        self.val_evaluator.run(self.valid_loader, max_epochs=1)
        self.iter += self.val_evaluator.state.iteration

    def run_test(self):
        self.test_evaluator.run(self.test_loader, max_epochs=1)

    def log_train_val(self, engine):
        lr = self.optimizer.param_groups[0]['lr']
        e = self.trainer.state.epoch
        n = self.trainer.state.max_epochs
        i = engine.state.iteration
        massage = f'Model {self.name} epoch {e}/{n} : {i}, lr: {lr:.6f}'
        for k, v in engine.state.metrics.items():
            massage += f',  {k}: {v:.3f}'
        print(massage)

    def log_test(self):
        e = self.trainer.state.epoch
        n = self.trainer.state.max_epochs
        i = self.test_evaluator.state.iteration

        massage = f'Model {self.name} epoch {e}/{n} : {i},'
        massage_ = ''
        j = 1
        for k, v in self.test_evaluator.state.metrics.items():
            if k == 'test accuracy':
                massage += f' {k} : {v:.2f}'
                print(massage)
            elif k != 'test cm':
                if j % 2 != 0:
                    massage_ = f' {k} : {v:.2f},'
                    massage_ += (12 - (len(massage_) - 18)) * ' '
                else:
                    massage_ += f' {k} : {v:.2f}'
                    print(massage_)
                j += 1
        self.log_confusion_matrix()

    def score_function(self, engine):
        return engine.state.metrics['valid accuracy']

    def global_step_transform(self, *args, **kwargs):
        return self.iter + self.val_evaluator.state.iteration

    def train_fn(self, engine, batch):
        self.model.train()
        images, targets = batch
        images = images.to(self.device, dtype=torch.float)
        targets = targets.to(self.device, dtype=torch.long)
        preds = self.model(images)
        loss = self.loss(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        return loss.item()

    def valid_fn(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            images, targets = batch
            images = images.to(self.device, dtype=torch.float)
            targets = targets.to(self.device, dtype=torch.long)
            preds = self.model(images)
            loss = self.loss(preds, targets)

        return preds, targets, loss.item()

    def log_confusion_matrix(self):
        cm = self.test_evaluator.state.metrics['test cm']
        cm = cm.numpy()
        cm = cm.astype(int)
        size = 8 if len(self.class_names) < 24 else len(self.class_names) // 3
        fig, ax = plt.subplots(figsize=(size, size))
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt="d")
        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(self.class_names, rotation=90)
        ax.yaxis.set_ticklabels(self.class_names, rotation=0)
        fig.tight_layout()
        plt.show()


