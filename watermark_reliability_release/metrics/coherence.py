"""
Adapted from the eval script in https://github.com/XiangLi1999/ContrastiveDecoding
requirement: pip install simcse (https://github.com/princeton-nlp/SimCSE)
"""
import numpy as np

from simcse import SimCSE

def get_coherence_score(prefix_text, generated_text, 
                        model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    
    print(len(prefix_text), len(generated_text))
    model = SimCSE(model_name)

    similarities = model.similarity(prefix_text, generated_text)
    similarities = np.array(similarities)
    coherence_score = similarities.trace() / len(similarities) 

    return coherence_score
