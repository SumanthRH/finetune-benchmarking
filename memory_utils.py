from transformers import AutoConfig

def get_activation_memory(model_config: AutoConfig, num_bytes: int =4, batch_size: int = 1, sequence_length:int = 512, grad_checkpointing=False, correction=True):
    b = batch_size
    N = num_bytes
    L = model_config.num_hidden_layers
    h = model_config.hidden_size
    a = model_config.num_attention_heads
    s = sequence_length
    if grad_checkpointing:
        activation_mem =  N*L*s*b*h # add temp memory + checkpointed mem # 
        if correction:
            activation_mem += (16*N+2)*s*b*h + (2*N+1)*a*s*s*b
    else:
        activation_mem = L*((16*N+2)*s*b*h + (2*N+1)*a*s*s*b)
    return activation_mem//(1024**2)
