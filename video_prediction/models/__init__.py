'''from .savp_model import BaseVideoPredictionModel

def get_model_class(model):
    model_mappings = {
        'ground_truth': 'GroundTruthVideoPredictionModel',
        'repeat': 'RepeatVideoPredictionModel',
        'savp': 'SAVPVideoPredictionModel',
        'dna': 'DNAVideoPredictionModel',
        'sna': 'SNAVideoPredictionModel',
        'sv2p': 'SV2PVideoPredictionModel',
    }
    model_class = model_mappings.get(model, model)
    model_class = globals().get(model_class)
    if model_class is None or not issubclass(model_class, BaseVideoPredictionModel):
        raise ValueError('Invalid model %s' % model)
    return model_class'''
