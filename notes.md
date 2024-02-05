# Notes
These are some notes I've taken while trying to get into the finer details of memory consumption during training. THe main goal is to see if we can quantify different overheads we see during training. 

# Kernels memory 
I've been trying to udnerstand how much memory gpu kernels take up. In  HF's blog, they mention that this init can be 1-2GB. I've noticed about 300MB of utilization from their code on RTX 3090 (workstation) and Tesla V100 (Colab). For an A100 this was about 500MB on Colab. 

# Model memory
Is there a difference in memory for model weights across different GPUs (like a mysterious overhead)? I think not. Between V100 and A100, I saw no difference (barring a few MBs) for a BERT model. 


# Gradient accumulation
On RTX 3090, with gradient accumulation, my calculations yielded ~ 7GB for BERT-Large, 1 batch size, 512 sequence length, float precision. This is different from the 8.3 GB observed memory utilization. For standard training, this gap was lesser. Why? temporary buffers?


# Mixed Precision
With FP16/ BF16, mixed precision trainingsaves optimzer state in fp32. Accumulation is done in FP32 as well for MMA. One can do pure Bf16 training but with the introduction of a special summation method like Kahan summation. Reference PR: https://github.com/pytorch/torchdistx/pull/52 

To understand this, first, we need to go back to floating point arithmetic. With floating points, you can imagine the datatype to be able to represent only numbers with a fixed length (in terms of number of digits), and the decimal place can "float" and be placed in any position. This means that when you're adding a large number with a small number, you will end up losing precision - you will lose the lower order bits of the small number. Pulling an example from wikipedia, using 7-digit significand decimal arithmetic:

1000.52 + 3.14159 = 1003.66159 => 10003.66 

Thus, to preserve the lower order bits, you need a summation method that will keep track of the lost lower order bits. One classic example is Kahan summation. (A modified version of this is used in Python's `sum()` method!)

So, for weight updates as well, in a pure BF16 setting, you can have the same issue with loss of precision after a number of training steps. Thus, you need a special summation method like Kahan summation to keep track of the error and add it back later. (this would still be approximate, of course) 

## More on Half-precision

When you have MMA ops, the multiplication typically happens block wise -. This means that at each step a thread will coompute a partial sum of the full dot product you need for an entry in the output matrix. WIth BF16/ FP16, this multiplication happens in the half-precision format, but the accumulation is in FP32. For example, you might multiply a (2x4) and a (4x2) matrix, which happens in BF16, but the result gets added to a (2x2) FP32 matrix.