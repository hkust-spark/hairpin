# Markov Model for FEC rate optimization and calculation

#### 1) Markov model without FEC decode delay
To get best FEC rate, run ```python3 fec_w_rtx.old.py --func best_beta```\
To get expected miss rate and bandwidth loss given certain network status, run ```python3 fec_w_rtx.old.py --func model```
#### 2) Markov model with FEC decode delay
To get best FEC rate given certain network status, run ```python3 fec_w_rtx.py --func best```\
To plot best FEC rate under different packet loss rate, run ```python3 fec_w_rtx.py --func plot```\
To get expected miss rate and bandwidth loss given certain network status, run ```python3 fec_w_rtx.py --func model```
#### 3) Generate table of best FEC rate
1. Run ```python3 fec_w_rtx.py --func offline_data``` and generate ```block_results_coeff.pkl/txt``` and ```model_result_coeff.pkl/txt```. It could take a long while.\
2. Run ```python3 merge_results.py``` and a FEC rate table written in C language ```fec-array_coeff.c/h``` will be generated.

If you need to distribute the workload to multiple servers, use ```distribute_workload.sh``` to automatically assign workload and distribute the python scripts to multiple hosts . But you need to run the python script on those hosts manually.\
Use ```collect_results.sh``` to automatically collect best FEC rate results from multiple servers, merge them into one big result file, generate a program interface in C language and copy them into the NS3 emulator we use.