import torch
import pathlib
import utils
import img_utils
import gvision_utils
import tensorflow as tf
import numpy as np
import math
import utils
from torchvision import models as torch_models
from torch.nn import DataParallel
from madry_mnist.model import Model as madry_model_mnist
from madry_cifar10.model import Model as madry_model_cifar10
from logit_pairing.models import LeNet as lp_model_mnist, ResNet20_v2 as lp_model_cifar10
from post_avg.postAveragedModels import pa_resnet110_config1 as post_avg_cifar10_resnet
from post_avg.postAveragedModels import pa_resnet152_config1 as post_avg_imagenet_resnet


class Model:
    def __init__(self, batch_size, gpu_memory):
        self.batch_size = batch_size
        self.gpu_memory = gpu_memory

    def predict(self, x):
        raise NotImplementedError('use ModelTF or ModelPT')

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = utils.softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()


class ImgSaveWrapper(Model):

    def __init__(self, model):
        self.model = model
        self.counter = 0

    def predict(self, x):
        logits = self.model.predict(x)

        img = img_utils.convert_to_pillow(x[0])
        img = img_utils.write_text_to_img(img, f"Iter: {self.counter}")
        img.save(f"output/saved_img_{self.counter}.png")

        self.counter += 1

        return logits


class GVisionModel:
    """Returns 2 simulated logits corresponding to correct class and incorrect class, such that attack has signal for optimization"""

    def __init__(self, correct_labelset=['cat'], exp_name="saved_img", save_location="output", loss_margin=1):
        self.labelset = correct_labelset
        self.save_location = save_location
        self.exp_name = exp_name
        self.counter = 0
        self.loss_margin = loss_margin
        self.best_loss = np.inf

        pathlib.Path(save_location).mkdir(parents=True, exist_ok=True)

        with open(f"{save_location}/labels.txt", "w") as f:
            f.write(", ".join(correct_labelset))

    def predict(self, x):
        if x.shape[0] != 1:
            raise AssertionError("Batch size must be 1")

        x = x[0]
        results = gvision_utils.gvision_classify_numpy(x)
        print(str(results))

        self.counter += 1

        logits = self._compute_logits(results)

        current_loss = self.loss(y=None, logits=logits)
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            img = img_utils.convert_to_pillow(x)
            img = img_utils.write_text_to_img(img, f"Iter: {self.counter}\n{results}")
            img.save(f"{self.save_location}/{self.exp_name}_{self.counter}.png")

        return logits


    def _compute_logits(self, results):
        """Confidence score of the correct and 2nd best class"""
        matching_results = results.match(self.labelset)
        other_results = results.match(self.labelset, inverse=True)


        print("Matching results:")
        print(matching_results)
        

        print("Matching results:")
        print(other_results)


        print(f"Top label: {matching_results.top_label} - {matching_results.top_score}")
        print(f"2nd best: {other_results.top_label} - {other_results.top_score}")

        print(f"Logits: [{matching_results.top_score}, {other_results.top_score}]")
        return np.array([[matching_results.top_score, other_results.top_score]])

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """

        if logits.shape[0] != 1:
            raise AssertionError("Batch size must be 1")
        if targeted:
            raise AssertionError("Targeted attack should be done by modifying the logit computation procedure")
        if loss_type != 'margin_loss':
            raise AssertionError("Gvision model only supports margin loss") 

        correct, second_best = logits[0]
        return np.array([correct - second_best + self.loss_margin])

class ModelTF(Model):
    """
    Wrapper class around TensorFlow models.

    In order to incorporate a new model, one has to ensure that self.model has a TF variable `logits`,
    and that the preprocessing of the inputs is done correctly (e.g. subtracting the mean and dividing over the
    standard deviation).
    """
    def __init__(self, model_name, batch_size, gpu_memory):
        super().__init__(batch_size, gpu_memory)
        model_folder = model_path_dict[model_name]
        model_file = tf.train.latest_checkpoint(model_folder)
        self.model = model_class_dict[model_name]()
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_file = model_file
        if 'logits' not in self.model.__dict__:
            self.model.logits = self.model.pre_softmax

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        tf.train.Saver().restore(self.sess, model_file)

    def predict(self, x):
        if 'mnist' in self.model_name:
            shape = self.model.x_input.shape[1:].as_list()
            x = np.reshape(x, [-1, *shape])
        elif 'cifar10' in self.model_name:
            x = np.transpose(x, axes=[0, 2, 3, 1])

        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        for i in range(n_batches):
            x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
            logits = self.sess.run(self.model.logits, feed_dict={self.model.x_input: x_batch})
            logits_list.append(logits)
        
        print(logits_list)
        logits = np.vstack(logits_list)
        return logits


class ModelPT(Model):
    """
    Wrapper class around PyTorch models.

    In order to incorporate a new model, one has to ensure that self.model is a callable object that returns logits,
    and that the preprocessing of the inputs is done correctly (e.g. subtracting the mean and dividing over the
    standard deviation).
    """
    def __init__(self, model_name, batch_size, gpu_memory):
        super().__init__(batch_size, gpu_memory)
        if model_name in ['pt_vgg', 'pt_resnet', 'pt_inception', 'pt_densenet']:
            model = model_class_dict[model_name](pretrained=True)
            self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
            self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
            model = DataParallel(model.cuda())
        else:
            model = model_class_dict[model_name]()
            if model_name in ['pt_post_avg_cifar10', 'pt_post_avg_imagenet']:
                # checkpoint = torch.load(model_path_dict[model_name])
                self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
                self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
            else:
                model = DataParallel(model).cuda()
                checkpoint = torch.load(model_path_dict[model_name] + '.pth')
                self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
                self.std = np.reshape([0.225, 0.225, 0.225], [1, 3, 1, 1])
                model.load_state_dict(checkpoint)
                model.float()
        self.mean, self.std = self.mean.astype(np.float32), self.std.astype(np.float32)

        model.eval()
        self.model = model

    def predict(self, x):
        x = (x - self.mean) / self.std
        x = x.astype(np.float32)

        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        with torch.no_grad():  # otherwise consumes too much memory and leads to a slowdown
            for i in range(n_batches):
                x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch, device=torch.device('cuda'))
                logits = self.model(x_batch_torch).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits


model_path_dict = {'madry_mnist_robust': 'madry_mnist/models/robust',
                   'madry_cifar10_robust': 'madry_cifar10/models/robust',
                   'clp_mnist': 'logit_pairing/models/clp_mnist',
                   'lsq_mnist': 'logit_pairing/models/lsq_mnist',
                   'clp_cifar10': 'logit_pairing/models/clp_cifar10',
                   'lsq_cifar10': 'logit_pairing/models/lsq_cifar10',
                   'pt_post_avg_cifar10': 'post_avg/trainedModel/resnet110.th'
                   }
model_class_dict = {'pt_vgg': torch_models.vgg16_bn,
                    'pt_resnet': torch_models.resnet50,
                    'pt_inception': torch_models.inception_v3,
                    'pt_densenet': torch_models.densenet121,
                    'madry_mnist_robust': madry_model_mnist,
                    'madry_cifar10_robust': madry_model_cifar10,
                    'clp_mnist': lp_model_mnist,
                    'lsq_mnist': lp_model_mnist,
                    'clp_cifar10': lp_model_cifar10,
                    'lsq_cifar10': lp_model_cifar10,
                    'pt_post_avg_cifar10': post_avg_cifar10_resnet,
                    'pt_post_avg_imagenet': post_avg_imagenet_resnet,
                    'gvision': GVisionModel
                    }
all_model_names = list(model_class_dict.keys())

