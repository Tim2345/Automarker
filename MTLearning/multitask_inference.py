
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import json
import torch
from collections import OrderedDict
import os
import transformers
import pickle
from tqdm import tqdm


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


class MultitaskInference(object):

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

        if not hasattr(model, 'dataset'):
            model.dataset = MTLearningDataset

        if not hasattr(model, 'data_loader'):
            model.dataloader = DataLoader

        model.device = check_cuda()

        return model

    def __init__(self, tokenizer, base_model, classifiers, dataset=MTLearningDataset, dataloader=DataLoader):
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.classifiers = classifiers
        self.dataset = dataset
        self.dataloader = dataloader

    def encodings_to_loader(self, encodings, n_samples, batch_size):
        texts_dataset = self.dataset(encodings, n_samples=n_samples)
        texts_dataloader = self.dataloader(texts_dataset, batch_size=batch_size)

        return texts_dataloader

    def tokenize_texts(self, texts, tasks_dict, max_len):
        texts = list(texts)

        encodings_dict = {}

        encodings_dict['text'] = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding='max_length'
        )
        print('tokenized tasks')

        if 'sentences' in tasks_dict.values():
            import spacy

            nlp = spacy.load('en_core_web_sm')

            sentence_encodings = []
            for text in texts:
                doc = nlp(text)
                sentences = [sent.text for sent in doc.sents]
                sentence_encodings.append(
                    self.tokenizer(
                        sentences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True)
                )

            encodings_dict['sentences'] = sentence_encodings

        return encodings_dict


    def infer_base_model(self, base_model_loader):
        self.base_model.to(self.device)

        base_model_outputs = []
        with torch.no_grad():
            for batch in tqdm(base_model_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                out = self.base_model(input_ids, attention_mask)
                base_model_outputs.extend(out.last_hidden_state)

        base_model_outputs = torch.stack(base_model_outputs)

        return base_model_outputs


    def infer_classifier(self, classifier_loader, task):

        self.classifiers[task].to(self.device)

        predictions = []
        with torch.no_grad():
            for batch in classifier_loader:
                out = self.classifiers[task](batch)
                predictions.extend(out)

        predictions = torch.stack(predictions)

        return predictions

    def predict(self, texts, tasks_dict, max_len, text_batch_size=25):
        # tokenize texts
        encodings_dict = self.tokenize_texts(texts, tasks_dict, max_len)
        print('ENCODED!!!')


        base_model_loader = self.encodings_to_loader(
            encodings_dict['text'],
            n_samples=len(encodings_dict['text'].encodings),
            batch_size=text_batch_size
        )
        print('loader created')

        base_model_text_out = self.infer_base_model(base_model_loader)


        print('base model text predictions made')

        if 'sentences' in tasks_dict.values():
            ## get base model predictions for text level encodings
            base_model_sentence_out = []
            for sent_encodings in encodings_dict['sentences']:

                base_model_loader = self.encodings_to_loader(
                    sent_encodings,
                    n_samples=len(sent_encodings['input_ids']),
                    batch_size=len(sent_encodings['input_ids'])
                )

                out_temp = self.infer_base_model(base_model_loader)
                base_model_sentence_out.append(out_temp)


        print('base model sentence predictions made')

        #### make classifications

        predictions_dict = dict()
        # get predictions
        for task in self.task_names:
            if tasks_dict[task] == 'sentences':
                sentence_preds = []

                for text_sents in base_model_sentence_out:
                    classifier_out_temp = self.infer_classifier(
                        self.dataloader(text_sents, batch_size=len(text_sents)),task
                    )

                    sentence_preds.append(classifier_out_temp)

                predictions_dict[task] = sentence_preds
                print('sentence_classifications made')

            else:
                task_preds = self.infer_classifier(
                    self.dataloader(base_model_text_out, batch_size=text_batch_size), task
                )
                predictions_dict[task] = task_preds

        return predictions_dict




