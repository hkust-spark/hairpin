#!/bin/bash

current_dir=$(pwd)

# ns3 version 3.33
ns3_ver="3.33"
ns3_folder="ns-allinone-${ns3_ver}"

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

ns3_root="${current_dir}/${ns3_folder}/ns-${ns3_ver}"
ns3_src="${ns3_root}/src"
ns3_scratch="${ns3_root}/scratch"
app_folder="bitrate-ctrl"
scratch_folder="scratch"
root_folder="ns3_root"
# creat soft link
if [ ! -d "${current_dir}/${app_folder}" ]
then
    echo "${app_folder} does not exist!"
    return
else
    # if soft link already exists, delete it
    if [ -d "${ns3_src}/${app_folder}" ]
    then
        rm -rf ${ns3_src}/${app_folder}
    fi
    echo "Linking all files..."
    # linking ./bitrate-ctrl
    ln -s -v ${current_dir}/${app_folder} ${ns3_src}/${app_folder}
    # linking ./scratch
    ln -s -f -v ${current_dir}/${scratch_folder}/* ${ns3_scratch}
    # linking ./ns3_root
    ln -s -f -v ${current_dir}/${root_folder}/* ${ns3_root}
fi

# compile (opitonal)
echo "Compiling ns3..."
cd ${current_dir}/${ns3_folder}
./build.py

echo "Enviorment set!"

