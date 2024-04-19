# import numpy as np
#
# from transformers import BertTokenizerFast
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# tokens = tokenizer("It is chair, that you sit in.", max_length=6, truncation=True, padding="max_length")
# print(len(tokens['input_ids']))
# print(tokens)
# tokens = tokenizer("It is chair, that you sit in.", max_length=10, truncation=True, padding="max_length")
# print(len(tokens['input_ids']))
# print(tokens)

def create_dependency_graph_by_spacy(sentence: str, spacy_nlp):
    # https://spacy.io/docs/usage/processing-text
    document = spacy_nlp(sentence)
    seq_len = len([token for token in document])
    print(seq_len)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')

    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
    return matrix

# import spacy
# from spacy import displacy
#
# chair = 'It is chair, that you sit in.'
#
# nlp=spacy.load('en_core_web_sm')
# graph = create_dependency_graph_by_spacy(chair, nlp)
# print(graph)
# #displacy.render(nlp(text1), jupyter=True)
# displacy.serve(nlp(chair), style="dep")

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("chair.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a kitten with blue eyes, sitting on a white surface",
                      "chair covered with cushion, easy cofort", "a chair"]).to(device)
print(text)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)  #b, 512
#     text_features = model.encode_text(text)  # b, 512
#     print(image_features.shape)
#     print(text_features.shape)
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]