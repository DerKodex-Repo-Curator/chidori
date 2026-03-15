---
title:   'Taints and Tolerates'
sidebar_position: 4
---
# Taints and Tolerates

Taints are applied to Agents to repel certain workloads unless they explicitly tolerate the taint. This feature is useful for reserving specialized hardware or ensuring certain workloads only run on specific Agents. 

## Practical Examples 

1. **Specialized Hardware:** 

    - **Taint**: An Agent with a high-end GPU might have a taint:
        ```powershell
        expensive-gpu=true 
        ```

    - **Usage**: To run on this Agent, a session must specify:
    
        ```powershell
        --tolerates expensive-gpu=true 
        ```

 

2. **Compliance Requirements:**

    - **Taint**: Agents meeting certain compliance standards might have a taint:
        ```powershell
        hipaa-compliant=required
        ```

    - **Usage**: Applications handling sensitive medical data would use:
        ```powershell
        --tolerates hipaa-compliant=required
        ```

 

3. **Experimental Features:**

    - **Taint**: Agents running beta software might have a taint:
        ```powershell
        beta-features=true
        ``` 

    - **Usage**: Developers testing new features would use:
        ```powershell
        --tolerates beta-features=true
        ```


## Example Commands 

- **Setting a taint on an Agent**: 

    ```powershell
    juice agent run --taints expensive-gpu=true 
    ```

- **Creating a session that tolerates the taint**:

    ```powershell
    juice run --tolerates expensive-gpu=true my_resource_intensive_app.py 
    ```

 