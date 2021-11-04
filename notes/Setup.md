-> used anaconda 3

https://cs231n.github.io/setup-instructions/
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Set up env:  
-> open Anaconda Powershell Prompt  
-> go to partition of project: ```e:```  
-> go to project: ```cd projects/cs231n```  
-> create new env: ```conda create -n cs231n python=3.7```  
-> enter env: ```conda activate cs231n```  
-> exit env: ```conda deactivate (opt: cs231n)``` or just exit terminal  
-> while in env, test python: ```python --version```  

---> also still have to install pytorch and matplotlib in conda environment

Install Packages:  
-> go to assignment folder ```cd assigment1```  
-> install required packages: ```pip install -r requirements.txt```  