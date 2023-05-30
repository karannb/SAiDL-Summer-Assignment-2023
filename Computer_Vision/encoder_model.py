#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 19:14:37 2023

@author: karan_bania
"""

'''
Encoder Using Clip.
'''

import torch
#import numpy as np
import torch.nn as nn
#import torch.nn.Functional as F
#import PIL as Image 
#from torchvision import transforms
#import clip
from transformers import CLIPModel, AutoTokenizer, AutoProcessor, CLIPTextModel, CLIPVisionModelWithProjection

device = "cuda" if torch.cuda.is_available() else "cpu"
        
model_id = "openai/clip-vit-base-patch32"

class Encoder(nn.Module):
    
    def __init__(self, model_id = model_id):
        super(Encoder, self).__init__()

        self.model = CLIPModel.from_pretrained(model_id)
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.preprocess = AutoProcessor.from_pretrained(model_id)#, do_center_crop=False)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
        
        self.text_encoder = CLIPTextModel.from_pretrained(model_id)

        #un-trainable
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id,
                                                                           output_hidden_states = True)
        
        #un-trainable
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        #self.image_encoder = self.image_encoder(output_hidden_states = True)
        
    def textForward(self, prompt):
        tokenized = self.tokenizer([prompt], padding=True, return_tensors='pt')
        prompt_embedding = self.text_encoder(**tokenized)

        return prompt_embedding

    def visualForward(self, image):
        preprocessed = self.preprocess(images=image, return_tensors='pt')
        image_embedding = self.image_encoder(**preprocessed)

        return image_embedding

        
    def forward(self, image, prompt, layers = [8, 9, 10, 11]):

        text_op = self.textForward(prompt)
        image_op_temp = self.visualForward(image)
        text_encoding = text_op[1]
        image_encoding = image_op_temp[0]
        mid_layers = []
        for i in range(len(layers)):
            mid_layers.append(image_op_temp['hidden_states'][layers[i]])
        
        #encoder_out = torch.stack((text_encoding, image_encoding), dim = 0)
        
        return text_encoding, image_encoding, mid_layers