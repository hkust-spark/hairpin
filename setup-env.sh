#!/bin/bash

current_dir=$(pwd)

# ns3 version 3.33
ns3_ver="3.33"
ns3_folder="ns-allinone-${ns3_ver}"

echo "syncing the ns3-sparkrtc submodule..."

git submodule update --init --recursive

# download ns3 to current dir
if [ ! -d "${current_dir}/${ns3_folder}" ]
then
    ns3_file_name="ns-allinone-${ns3_ver}.tar.bz2"
    url="https://www.nsnam.org/releases/${ns3_file_name}"
    if [ ! -f "${current_dir}/${ns3_file_name}" ]
    then
        echo "Downloading NS3-${ns3_ver}..."
        wget --show-progress --quiet $url
    fi
    # unzip
    echo "Unzipping ${ns3_file_name}..."
    tar xjf ${ns3_file_name}
fi

# Copy Hairpin weights
echo "Copying hairpin weights..."
cp ./model/beta-array-rtx*.bin ./sparkrtc/model/fec/

ns3_root="${current_dir}/${ns3_folder}/ns-${ns3_ver}"
ns3_src="${ns3_root}/src"
app_folder="sparkrtc"
root_folder="ns3-scripts"
# creat soft link
if [ ! -d "${current_dir}/${app_folder}" ]
then
    echo "${app_folder} does not exist!"
    return
else
    echo "Linking all files..."
    # if soft link already exists, delete it
    if [ -d "${ns3_src}/${app_folder}" ]
    then
        rm -rf ${ns3_src}/${app_folder}
    fi
    # linking ./ns3-sparkrtc
    ln -s -f -v ${current_dir}/${app_folder} ${ns3_src}/${app_folder}
    # linking ./ns3_root
    ln -s -f -v ${current_dir}/${root_folder}/* ${ns3_root}
fi

# compile
echo "Compiling ns3..."
cd ${ns3_root}
LDFLAGS="-lboost_filesystem -lboost_system" ./waf configure --cxx-standard=-std=c++17 --disable-python --enable-examples
./waf

echo "Enviorment set!"

