
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import json
import torch
from collections import OrderedDict
import os
import transformers
import pickle
from tqdm import tqdm

from Automarker.scripts.data_prep.train_test_set import augmented_data, cleaned_data


class MTLearningDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, n_samples):
        self.encodings = encodings
        self.n_samples = n_samples

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.n_samples



def get_model_config_dict(checkpoint_dir):
    model_dict = dict()

    for mod in os.walk(checkpoint_dir):
        model_names = [name for name in os.listdir(mod[0]) if name.endswith('.bin')]
        if any(model_names):
            task_name = mod[0].split('/')[-1]
            f = open(mod[0] + '/config.json')
            config = json.load(f)
            f.close()
            config['PATH'] = mod[0]
            model_dict[task_name] = config

    return model_dict


def load_model(model_type, model_path):
    model = object()
    if 'SequenceClassification' in model_type:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

    elif 'TokenClassification' in model_type:
        model = transformers.AutoModelForTokenClassification.from_pretrained(model_path)

    else:
        raise ValueError('Unable to determine head type to load.')

    print("Loaded '{}' from '{}'".format(model.config.architectures[0], model_path))
    return model

def check_cuda():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(
            "Using {} for inference. Assign other device to 'self.device' with 'torch.cuda.get_device_name()'.".format(gpu))
        return 'cuda'
    else:
        print("No GPU found. Using CPU for inference.")

        return 'cpu'


class MultitaskInference:

    @property
    def task_names(self):
        return list(self.classifiers.keys())

    @classmethod
    def decompose_from_checkpoint(cls, checkpoint_dir, save_path=None):
        if save_path == None:
            save_path = checkpoint_dir


        model_dict = get_model_config_dict(checkpoint_dir)

        loaded_models = {
            model_name: load_model(model_dict[model_name]['architectures'][0], model_dict[model_name]['PATH'])
            for model_name, model_path
            in model_dict.items()
        }

        base_model = loaded_models[list(model_dict.keys())[0]].base_model

        classifiers = {task_name: loaded_model.classifier for task_name, loaded_model in loaded_models.items()}

        tokenizer = AutoTokenizer.from_pretrained(base_model.config.pretrained_model_type)

        model = cls(tokenizer, base_model, classifiers)

        model_name = '_'.join([val[:2] for val in classifiers.keys()])

        # pickle file
        with open('{}/MultitaskModel_{}.pkl'.format(save_path, model_name), 'wb') as f:
            pickle.dump(model, f)

        print("Model decomposed and pickled in '{}'".format(save_path))
        print("Base model encoder: {}".format(base_model.config.pretrained_model_type))
        print("Model tasks: {}".format(model.task_names))

        model.device = check_cuda()

        return model

    @staticmethod
    def load_multitask(model_dir):
        with open(model_dir, 'rb') as f:
            model = pickle.load(f)

        if not isinstance(model, MultitaskInference):
            raise RuntimeWarning("Object not an instance of 'MultitaskInference' class.")

        if not hasattr(model, 'tokenizer'):
            print("\033[0;31mLoaded object has no tokenizer. One must be assigned before inference.\033[0m")

        if not hasattr(model, 'base_model'):
            print("\033[0;31mLoaded object has no base model. One must be assigned before inference.\033[0m")

        if not hasattr(model, 'classifiers'):
            print("\033[0;31mLoaded object has no classifiers. At least one must be assigned before inference.\033[0m")

        model.device = check_cuda()

        return model

    def __init__(self, tokenizer, base_model, classifiers):
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.classifiers = classifiers


    def infer_base_model(self, base_model_loader):
        self.base_model.to(self.device)

        base_model_outputs = []
        with torch.no_grad():
            for batch in tqdm(base_model_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                out = model.base_model(input_ids, attention_mask)
                base_model_outputs.extend(out.last_hidden_state)

        base_model_outputs = torch.stack(base_model_outputs)

        return base_model_outputs


    def infer_classifier(self, task, classifier_loader):

        model.classifiers[task].to(model.device)

        predictions = []
        with torch.no_grad():
            for batch in tqdm(classifier_loader):
                out = self.classifiers[task](batch)
                predictions.extend(out)

        predictions = torch.stack(predictions)

        return predictions

###load and run predictions
# load all models from checkpoint

checkpoint_dir = './multitask_model_adjusted_practice_175/checkpoint-2709'
texts = list(cleaned_data['ANSWER'].sample(100))
task = 'automarker'

text = '''

texts are really nice things to consider when you are making the best of something interesting.
I never really knew what to think of the man who sat by the side of the road until he mentioned to me that there was 
something odd and dispassionate about his face. he stood byt he side of the road and decided to carry on walking. to where, 
no body knew. the only thing he wanted to was consider the thought of not even being htere any more.

'''

#model = MultitaskInference.decompose_from_checkpoint(checkpoint_dir)

batch_size = 25
max_len = 200

model = MultitaskInference.load_multitask(checkpoint_dir+'/MultitaskModel_co_au.pkl')

texts_encoded = model.tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_len, padding='max_length')
texts_dataset = MTLearningDataset(texts_encoded,  n_samples=len(texts))
texts_dataloader = DataLoader(texts_dataset, batch_size=batch_size)

base_model_out = model.infer_base_model(texts_dataloader)

classifier_loader = DataLoader(base_model_out)

final_predictions = model.infer_classifier(task, classifier_loader)


### data
model.base_model.to(model.device)

base_model_outputs = []
with torch.no_grad():
    for batch in tqdm(texts_dataloader):

        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        out = model.base_model(input_ids, attention_mask)
        base_model_outputs.extend(out.last_hidden_state)

base_model_outputs = torch.stack(base_model_outputs)

base_model_outputs[0].size()
base_model_outputs[:2]
model.classifiers[task].to(model.device)
model.classifiers[task](base_model_outputs[:2])

head_loader = DataLoader(base_model_outputs, batch_size=batch_size)

predictions = []
with torch.no_grad():
    for batch in tqdm(head_loader):

        out = model.classifiers[task](batch)
        predictions.extend(out)










