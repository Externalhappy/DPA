import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

@torch.no_grad()
def build_memory(args, model, dataset_name, data_loader, len_dataset, features_shape):
    model.eval()
    model_name = args.clip_model
    model_name = model_name.replace("/", "").replace("-", "")
    Path('center_embedding').mkdir(parents=True, exist_ok=True)
    if os.path.isfile(f'center_embedding/{model_name}{dataset_name}_embeddings.pt'):
        print('******** Loaded Already Saved center Embeddings *********')
        center = torch.load(f'center_embedding/{model_name}{dataset_name}_embeddings.pt')
        features = torch.load(f'center_embedding/{model_name}{dataset_name}_features.pt')
        labels = torch.load(f'center_embedding/{model_name}{dataset_name}_labels.pt')
        center = center.clone().detach()
        memory = {
            "features" : features,
            "labels" : labels,
        }
    else:
        print('******** No center Embeddings Found --- Saving New Embeddings *********')
        features = torch.zeros(len_dataset, features_shape)
        labels = -1*torch.ones(len_dataset, dtype=torch.long)
        total = 0.0
        total_correct = 0.0
        for _, batch in enumerate(tqdm(data_loader)):
            inputs, label, idx = batch
            image = inputs[0].to(args.device)
            with torch.no_grad():
                feats = model(image)
                logit = 100. * feats @ model.get_classifier().t()
            proba = torch.softmax(logit, dim=-1)
            pseudo_targets = torch.argmax(proba, dim=-1)
            features[idx] = feats.detach().cpu()
            labels[idx] = pseudo_targets.detach().cpu()
            total += idx.shape[0]
            total_correct += (label == pseudo_targets.detach().cpu()).float().sum()
            del image, logit, feats, proba, pseudo_targets
        print(f'************** Accuracy = {(total_correct/total)*100:0.2f} **************')
        memory = {
            "features" : features,
            "labels" : labels,
        }
        center = get_center(args, memory, model.get_classifier())
        torch.save(center, f'center_embedding/{model_name}{dataset_name}_embeddings.pt')
        torch.save(features, f'center_embedding/{model_name}{dataset_name}_features.pt')
        torch.save(labels, f'center_embedding/{model_name}{dataset_name}_labels.pt')
    return center, memory


def get_center(args, memory, classifier):
    features = memory['features']
    labels = memory['labels']
    available_labels = torch.arange(args.nb_classes)
    class_means  = []
    for i in tqdm(available_labels):
        idx = (labels == i)
        samples_count = idx.float().sum().item()
        if samples_count > 0.0:
            feat = features[idx]
            class_emebdding = torch.mean(feat,dim=0)
            class_emebdding /= class_emebdding.norm()
        else :
            print(f'class [{i}] was not found as pseudo-labels and was replaces with txt center.')
            class_emebdding =  classifier[i].detach().cpu()
            class_emebdding /= class_emebdding.norm()
        class_means.append(class_emebdding)
    center = torch.stack(class_means, 0)
    return center

@torch.no_grad()
def refine_pseudoDA(train_config, probs_txt, prob_list, center, features):
    alpha = train_config['a']
    prob_avg = torch.stack(prob_list,dim=0).mean(0)
    probs_txt = probs_txt / prob_avg
    probs_txt = probs_txt / probs_txt.sum(dim=1, keepdim=True)   
    logit_centre  = 100.0 * features @ center.t()
    probs_centre  = F.softmax(logit_centre.float(), -1)
    probs = alpha*probs_centre + (1-alpha)*probs_txt
    return probs, probs_txt, probs_centre

@torch.no_grad()
def get_weights(probs_centre, labels):
    max_values, _ = torch.max(probs_centre, dim=1, keepdim=True)
    normalized_tensor = probs_centre / (max_values+1e-7)
    weights = torch.tensor([row[idx].item() for row, idx in zip(normalized_tensor, labels)])
    return weights


def get_center_v2(args, memory, classifier):
    features = memory['features']
    labels = memory['labels']
    available_labels = torch.arange(args.nb_classes)
    class_means  = []
    for i in tqdm(available_labels):
        idx = (labels == i)
        samples_count = idx.float().sum().item()
        if samples_count > 0.0:
            feat = features[idx] # (samples_count, 512)
            feat_norm = feat / feat.norm(dim=1, keepdim=True)
            cos_sim_affin = (torch.mm(feat_norm, feat_norm.t()).sum(0) - 1) / (int(samples_count) - 1 + 1e-7)
            sim_weight = F.softmax(cos_sim_affin / 0.2, dim=0)
            class_emebdding = torch.sum(sim_weight[:, None] * feat, dim=0) # Similarity weighted aggregation.
            class_emebdding /= class_emebdding.norm()
        else :
            print(f'class [{i}] was not found as pseudo-labels and was replaces with txt center.')
            class_emebdding =  classifier[i].detach().cpu()
            class_emebdding /= class_emebdding.norm()
        class_means.append(class_emebdding)
    center = torch.stack(class_means, 0)
    return center