We also provide a WebRTC-based implementation for Hairpin at [hairpin-webrtc](https://github.com/hkust-spark/hairpin-webrtc).

Hairpin is built on SparkRTC ns-3 library, which provides a simulation for the real-time video communication in ns-3.
It provides congestion control algorithms (GCC, NADA) and forward error correction baselines.
The main functionality has been implemented in the [ns3-sparkrtc](https://github.com/hkust-spark/ns3-sparkrtc) repository.
This repository is used to generate the parameters of Hairpin and run the experiments.

Please cite our paper if you use our simulator:
```
@inproceedings{nsdi2024hairpin,
  title={Hairpin: Rethinking Packet Loss Recovery in Edge-based Interactive Video Streaming},
  author={Meng, Zili and Kong, Xiao and Chen, Jing and Wang, Bo and Xu, Mingwei and Han, Rui and Liu, Honghao and Arun, Venkat and Hu, Hongxin and Wei, Xue},
  booktitle={Proc. USENIX NSDI},
  year={2024}
}
```

The packet can be parsed with the `sparkrtc.lua` in Wireshark.

## Install dependency
```bash
sudo apt install build-essential libboost-all-dev
```

## Fetch the latest ns-3 source files
```bash
./setup-env.sh
```

## Configure before building!
For ns-3.33 and below, to use c++14 and above features, should configure like this:
([More information: C++ standard configuration in build system (#352) · Issues · nsnam / ns-3-dev · GitLab](https://gitlab.com/nsnam/ns-3-dev/-/issues/352))
```bash
LDFLAGS="-lboost_filesystem -lboost_system" ./waf configure --cxx-standard=-std=c++17
```

## Run the code
To simply test if the codes can run, just run at `ns-allinone-3.33/ns-3.33`:
```bash
./waf --run "rtc-test --vary=1 --fecPolicy=hairpin"
```
When the process is running, the logs can be found at `ns-allinone-3.33/ns-3.33/logs/`.
Usually the program will finish in around 5 minutes.

## Reproduce the results
We use `generate_conf_*` to generate configuration files, where the files are later put into `*.conf` files.
For example, to run the hairpinone baseline in the WiFi traces, just run
```bash
python run_ns3.py --conf wifi-hairpinone.conf
```
`run_ns3.py` is a wrapper to parallelly run ns-3 experiments over different traces and in parallel.

If you want to run a single code for debugging, copy one line from the `conf` file and use ns-3 command line to run it.

## Process the results
The `process_results.py` is used to process the results and generate the data to plot figures in the paper.

## Generate parameters for other coefficient
```bash
cd model
python fec_w_rtx.py --func offline_data -coeff 1e+00
python merge_results.py --coeff 1e+00
```
Currently the coefficient only supports $10^n$.
