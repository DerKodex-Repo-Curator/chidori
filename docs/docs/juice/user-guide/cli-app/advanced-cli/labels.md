---
title:   'Labels and Match Labels'
sidebar_position: 3
---
# Labels and Match Labels

Labels in Juice are key-value pairs that can be attached to agents (GPU-hosting machines). They provide a flexible and powerful way to organize, categorize, and select agents based on various attributes. Match labels are used when creating sessions or running applications to ensure that the workload is assigned to an agent with the specified labels. 

 

## Applying Labels 

Labels are applied to agents when they are started or configured. Multiple labels can be applied to a single agent, and each label consists of a key and a value, separated by an equals sign. 

**Example**:

```powershell
gpu-type=a100 
```
 

## Using Match Labels 

- When creating a session or running an application, users can specify match labels to select agents with specific attributes. 

- The system will only consider agents that have **all** of the specified labels. 

 

## Practical Examples 

1. **GPU Type and Capacity:** 

    ```powershell
    juice agent run --labels gpu-type=a100,gpu-memory=80gb,cuda-version=11.5
    ```

    This command starts an agent, labeling it with an A100 GPU, 80GB of GPU memory, and CUDA version 11.5. Run the following command to use this agent:
    
    ```powershell
    juice run --match-labels gpu-type=a100,gpu-memory=80gb my_large_model_training.py
    ```
    This runs *my_large_model_training.py* on an agent with an A100 GPU and 80GB of memory.

---

2. **Software Environment and Version:**

    ```powershell
    juice agent run --labels env=pytorch,env-version=1.9,python-version=3.8
    ```
    
    This labels an agent with PyTorch 1.9 and Python 3.8. 

---

3. **Usage:**

    ```powershell
    juice run --match-labels env=pytorch,env-version=1.9 my_pytorch_script.py
    ```
    
    This ensures my_pytorch_script.py runs on an agent with PyTorch 1.9. 

---

4. **Location and Purpose:**
    
    ```powershell
    juice agent run --labels location=us-east,purpose=inference,network-speed=10gbps
    ```
    
    This labels an agent for inference tasks in the US East region with a 10Gbps network. 

---

5. **Usage:**

    ```powershell
    juice run --match-labels location=us-east,purpose=inference my_inference_service.py 
    ```

    This runs an inference service on an appropriate agent in the US East region. 

---

6. **Combining Multiple Criteria:**

    ```powershell
    juice run --match-labels gpu-type=v100,env=tensorflow,location=eu-west,purpose=training my_training_job.py 
    ```

    This complex example runs my_training_job.py on an agent that has a V100 GPU, TensorFlow installed, is located in Western Europe, and is designated for training purposes. 

 

## Best Practices for Using Labels 

- **Consistency**: Develop a consistent labeling scheme across your organization. 

 

- **Granularity**: Use labels that are specific enough to be useful but not so specific that they become cumbersome. 

 

- **Documentation**: Maintain documentation of your label schema for all users. 

 
- **Regular Review**: Periodically review and update your labeling strategy as your infrastructure and needs evolve. 

 

## Advanced Usage: Label Selectors 

Juice also supports more complex label selectors for advanced use cases. Some of these commands include: 

- **Multiple Values**: To match agents with either V100 or A100 GPUs 

    ```powershell
    juice run --match-labels "gpu-type in (v100,a100)"
    ```

- **Exclusions**: To avoid production environments 

    ```powershell
    juice run --match-labels "env!=production"
    ```
  
With effective use of labels and match labels, juice users can create a highly organized and efficient GPU resource allocation system to ensure that each workload runs on the most appropriate environment configuration.  