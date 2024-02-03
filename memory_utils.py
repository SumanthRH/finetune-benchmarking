from transformers import AutoConfig

def get_activation_memory(model_config: AutoConfig, num_bytes: int =4, batch_size: int = 1, sequence_length:int = 512):
    b = batch_size
    N = num_bytes
    L = model_config.num_hidden_layers
    h = model_config.hidden_size
    a = model_config.num_attention_heads
    s = sequence_length
    return L*((16*N+2)*s*b*h + (2*N+1)*a*s*s*b)
