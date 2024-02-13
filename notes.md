# Notes
These are some notes I've taken while trying to get into the finer details of memory consumption during training. THe main goal is to see if we can quantify different overheads we see during training. 

# Kernels memory 
I've been trying to udnerstand how much memory gpu kernels take up. In  HF's blog, they mention that this init can be 1-2GB. I've noticed about 300MB of utilization from their code on RTX 3090 (workstation) and Tesla V100 (Colab). For an A100 this was about 500MB on Colab. 
`CUDA_MODULE_LOADING=EAGER` forces pytorch to load all the cuda kernels at the start. Typically torch will lazily load this. 

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

## Mixed precision in practice
In practice, with `torch.autocast` and DeepSpeed, the full fp32 gradients are materialized. Consider what the mixed precision paper mentions -  weights, activations and gradients are in half-precision, and during the optimizer step, the gradients are converted to FP32. However, not materializing the full FP32 gradients and converting them on the fly would need to be an implementation in the optimizer. This is not possible with the usual torch optim APIs. Thus, at runtime, the full FP32 gradients would be stored in HBM. 

After you finish training, gradients are cleared, so these 4 bytes would contribute to the temporary memory allocated for the program.

# Overhead
The gpu performance equation has a latency term: 
T = T_mem + T_math + latency

This latency or overhead is the time spend doing everything else apart from fetching memory or doing compute. This can be time spent in the Python interpreter, time spent in Pytorch, etc. Pytorch executes cuda code asynchronously to overlap overhead with computations. This means that while the GPU is executing CUDA code, pytorch's dispatcher can dispatch the next set of cuda kernels to be executed. 

# Profiling

You can use pytorch's native profiler to profile GPU utlization. Note that GPU ulitization - the percentage of SMs/cores you're keeping active - is different from the "volatile GPU utilization" that you see on nvidia-smi. "volatile gpu utilization" is the percentage of time a CUDA kernel is active - this means that you can get 100% utlization even if you're just using 1 core. 

You can also use nsys to profile https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59 

# Baseline

WIth simple FP32 training with 4 batch size, 512 sequence length for BERT-Large:
```
{'train_runtime': 43.3639, 'train_samples_per_second': 11.807, 'train_steps_per_second': 2.952, 'train_loss': 0.009557262994349003, 'epoch': 1.0}       
Time: 43.36                                                                                                              
Samples/second: 11.81                                                                       
GPU memory occupied: 11844 MB.
```

# Gradient Accumulation
4 gradient accumulation steps. 

```
{'train_runtime': 56.0154, 'train_samples_per_second': 9.14, 'train_steps_per_second': 2.285, 'train_loss': 0.011208745650947094, 'epoch': 1.0}
Time: 56.02
Samples/second: 9.14
GPU memory occupied: 8330 MB.
```


# Tensor Cores?

Helpful answer from stack overflow on tensor cores:

GPUs have 5120 cuda cores where each core can perform up to 1 single precision multiply-accumulate operation (e.g. in fp32: x += y * z) per 1 GPU clock (e.g. Tesla V100 PCIe frequency is 1.38Gz).

Each tensor core perform operations on small matrices with size 4x4. Each tensor core can perform 1 matrix multiply-accumulate operation per 1 GPU clock. It multiplies two fp16 matrices 4x4 and adds the multiplication product fp32 matrix (size: 4x4) to accumulator (that is also fp32 4x4 matrix).

It is called mixed precision because input matrices are fp16 but multiplication result and accumulator are fp32 matrices.

Probably, the proper name would be just 4x4 matrix cores however NVIDIA marketing team decided to use "tensor cores".

# Fused Adam?
https://discuss.pytorch.org/t/fusedadam-optimizer-in-nvidia-amp-package/47544

> The Adam optimizer in Pytorch (like all Pytorch optimizers) carries out optimizer.step() by looping over parameters, and launching a series of kernels for each parameter. This can require hundreds of small launches that are mostly bound by CPU-side Python looping and kernel launch overhead, resulting in poor device utilization. Currently, the FusedAdam implementation in Apex flattens the parameters for the optimization step, then carries out the optimization step itself via a fused kernel that combines all the Adam operations. In this way, the loop over parameters as well as the internal series of Adam operations for each parameter are fused such that optimizer.step() requires only a few kernel launches.