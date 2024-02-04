# Notes
These are some notes I've taken while trying to get into the finer details of memory consumption during training. THe main goal is to see if we can quantify different overheads we see during training. 

# Kernels memory 
I've been trying to udnerstand how much memory gpu kernels take up. In  HF's blog, they mention that this init can be 1-2GB. I've noticed about 300MB of utilization from their code on RTX 3090 (workstation) and Tesla V100 (Colab). For an A100 this was about 500MB on Colab. 

# Model memory
Is there a difference in memory for model weights across different GPUs? I think not. Between V100 and A100, I saw no difference. 



