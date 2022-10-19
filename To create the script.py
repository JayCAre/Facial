#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision.all import *
import gradio as gr
learner = load_learner("facial_expression_class.pkl")
categories = ('happy','sad','scared', 'angry', 'surprised')

def classify_emotion(img):
    pred,idx,probs = learner.predict(img)
    return dict(zip(categories,map(float,probs)))

image= gr.inputs.Image(shape =(192,192))
label = gr.outputs.Label()
examples= ["download (1).jpg"]

intf = gr.Interface(fn=classify_emotion, inputs=image, outputs=label,examples=examples)
intf.launch(inline=False)

