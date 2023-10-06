## Install dependency
```bash
sudo apt install build-essential libboost-all-dev
```

## Fetch the latest ns-3 source files
```bash
cd emulation
./setup-env.sh
```

## Configure before building!
For ns-3.33 and below, to use c++14 and above features, should configure like this:
([More information: C++ standard configuration in build system (#352) · Issues · nsnam / ns-3-dev · GitLab](https://gitlab.com/nsnam/ns-3-dev/-/issues/352))
```bash
LDFLAGS="-lboost_filesystem -lboost_system" ./waf configure --cxx-standard=-std=c++17
```

## Run the code
We use `generate_conf_*` to generate configuration files, where the files are later put into `*.conf` files.
For example, to run the hairpinone baseline in the WiFi traces, just run
```bash
python run_ns3.py --conf wifi-hairpinone.conf
```
`run_ns3.py` is a wrapper to parallelly run ns-3 experiments over different traces.

If you want to run a single code for debugging, copy one line from the `conf` file and use ns-3 command line to run it.

## Process the results
The `process_results.py` is used to process the results and generate the data to plot figures in the paper.

## Generate parameters for other coefficient
```bash
cd simulation/code/model
python fec_w_rtx.py --func offline_data -coeff 1e+00
python merge_results.py --coeff 1e+00
```
Currently the coefficient only supports $10^n$.