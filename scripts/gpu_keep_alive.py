import torch
import time
import multiprocessing
from pynvml import *
from math import sqrt

# ------------------- parameters -------------------
# memory threshold (GB)
# when the available memory of the GPU is higher than this value, the sub-process will try to occupy it
MEMORY_THRESHOLD_GB = 20 
# the size of the tensor to occupy on each GPU (GB)
# note: this value will create two tensors of the same size, so the total occupied memory will be twice this value
# for example, setting to 15 will try to occupy 2 * 15 = 30GB of VRAM
TENSOR_SIZE_GB = 30
# --------------------------------------------------

def get_gpu_memory_usage(gpu_id):
    """get the current memory usage of the specified GPU (GB)"""
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**3

def occupy_gpu_memory(gpu_id):
    """
    create a Tensor on the specified GPU to occupy VRAM
    this is the core program to occupy VRAM
    """
    device_name = f'cuda:{gpu_id}'
    device = torch.device(device_name)
    
    # calculate the number of elements based on the desired GB size (float32 takes 4 bytes)
    num_elements = 44869*44869
    
    print(f"[GPU {gpu_id}] number of elements in the Tensor: {num_elements}")
    print(f"[GPU {gpu_id}] start allocating {TENSOR_SIZE_GB * 2} GB of VRAM on device {device_name}...")

    large_tensor1 = None
    large_tensor2 = None
    try:
        # create two large Tensors to occupy VRAM
        large_tensor1 = torch.empty(num_elements, dtype=torch.float32, device=device)
        large_tensor2 = torch.empty(num_elements, dtype=torch.float32, device=device)

        # fill the Tensor to ensure VRAM is actually allocated and used
        large_tensor1.fill_(1.0)  
        large_tensor2.fill_(2.0)  
        
        print(f"[GPU {gpu_id}] VRAM allocation successful. Enter infinite calculation loop to keep occupied...")
        # infinite calculation loop to prevent program exit and VRAM release
        while True:
            # perform a simple operation to keep GPU active
            size = int(sqrt(num_elements))
            matrix2 = large_tensor2.view(size, size)
            matrix1 = large_tensor1.view(size, size)
            torch.matmul(matrix1, matrix2) 
            time.sleep(1) # can add a short sleep to reduce CPU overhead of the loop

    except RuntimeError as e:
        print(f"[GPU {gpu_id}] error: {e}")
        # clean up GPU memory
        del large_tensor1
        del large_tensor2
        torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print(f"[GPU {gpu_id}] process interrupted.")

def manage_gpu_for_process(gpu_id):
    """
    manage the sub-process of a single GPU.
    it will continuously monitor the memory of the specified GPU, and call the occupy function when the memory is sufficient.
    """
    try:
        # each sub-process needs to initialize NVML independently
        nvmlInit()
        print(f"[GPU {gpu_id}] sub-process started, start monitoring...")
        
        while True:
            current_memory = get_gpu_memory_usage(gpu_id)
            print(f"[GPU {gpu_id}] current VRAM usage: {current_memory:.2f}GB")

            # if the current VRAM is less than the threshold, it means the VRAM is still idle, can start occupying
            if current_memory < MEMORY_THRESHOLD_GB:
                print(f"[GPU {gpu_id}] VRAM is less than {MEMORY_THRESHOLD_GB}GB, start executing occupy program...")
                # call the occupy function, this function is a dead loop, will run until an error or interruption
                occupy_gpu_memory(gpu_id)
                # if occupy_gpu_memory exits due to an error, print a message and wait for the next check
                print(f"[GPU {gpu_id}] occupy program exited, will check again in 5 seconds.")

            time.sleep(5)  # check memory usage every 5 seconds

    except KeyboardInterrupt:
        print(f"[GPU {gpu_id}] monitoring process interrupted.")
    finally:
        # each sub-process needs to close NVML independently before exiting
        nvmlShutdown()
        print(f"[GPU {gpu_id}] sub-process closed.")

if __name__ == "__main__":
    # ensure the code runs in the main module when using multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    try:
        # get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("error: no available GPUs detected.")
            exit()
            
        print(f"detected {num_gpus} GPUs. will start a management process for each GPU.")

        processes = []
        # create an independent process for each GPU
        for i in range(num_gpus):
            process = multiprocessing.Process(target=manage_gpu_for_process, args=(i,))
            processes.append(process)
            process.start()

        # wait for all sub-processes to end (in this script, the sub-processes are dead loops, so the main process will wait)
        for process in processes:
            process.join()

    except KeyboardInterrupt:
        print("\nmain program interrupted by user (Ctrl+C), terminating all sub-processes...")
        # when the user presses Ctrl+C, the main process will receive KeyboardInterrupt
        # join() will interrupt here, then the program will exit, and the sub-processes will also exit
    finally:
        print("main program exited.")


# import torch
# import time

# device = torch.device('cuda:0')

# num_elements = int(30 * 1024**3 / 4)    #62057
# # num_elements = int(20 * 1024**3 / 4)    #41569
# # num_elements = int(18 * 1024**3 / 4)    
# # num_elements = int(10 * 1024**3 / 4)    #21090

# try:
#     large_tensor1 = torch.empty(num_elements, dtype=torch.float32, device=device)
#     large_tensor2 = torch.empty(num_elements, dtype=torch.float32, device=device)
#     # print("Tensors allocated on GPU, occupying ~120GB of VRAM.")

#     large_tensor1.fill_(1.0)  
#     large_tensor2.fill_(2.0)  

#     while True:
#         large_tensor1.add_(large_tensor2)

# except RuntimeError as e:
#     print("Failed to allocate tensor on GPU or run computation. Error:", e)