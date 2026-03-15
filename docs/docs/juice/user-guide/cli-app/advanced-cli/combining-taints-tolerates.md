---
title:   'Combining Taints and Tolerates'
sidebar_position: 5
---
# Combining Taints and Tolerates

Labels and taints can be used together for fine-grained control over workload placement. 

 
## Example 

```powershell
juice run --match-labels gpu-type=a100,env=pytorch-2.0 --tolerates hipaa-compliant=required,expensive-gpu=true my_secure_ml_app.py 
```
:::info
This command would run *my_secure_ml_app.py* on an agent that: 
 
1. Has an A100 GPU 

2. Has PyTorch 2.0 installed 

3. Meets HIPAA compliance requirements 

4. Is explicitly allowed to run expensive workloads 

:::

Juice users can leverage these features to ensure their workloads are run on appropriate hardware and in the correct environments, balancing performance needs with resource management and compliance requirements.  