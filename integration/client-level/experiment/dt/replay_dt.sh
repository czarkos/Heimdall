#!/bin/bash

set -euo pipefail

# This script mirrors flashnet/replay.sh but uses the surrogate
# decision tree replayer binary (io_replayer_dt).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

user=""
original_device_index=""
devices_list=""
trace=""
output_dir=""
duration=""

while [ $# -gt 0 ]; do
  case "$1" in
    -user)
      user="$2" # to regain the access from the sudo user
      ;;
    -original_device_index)
      original_device_index="$2"
      ;;
    -devices_list)
      devices_list="$2"
      ;;
    -trace)
      trace="$2"
      ;;
    -output_dir)
      output_dir="$2"
      ;;
    -duration)
      duration="$2"
      ;;
    *)
      printf "ERROR: Invalid argument. Expected flags: -user -original_device_index -devices_list -trace -output_dir -duration\n"
      exit 1
  esac
  shift
  shift
done

if [ -z "${devices_list}" ] || [ -z "${trace}" ] || [ -z "${output_dir}" ] || [ -z "${duration}" ] || [ -z "${original_device_index}" ]; then
    echo "ERROR: Missing required arguments."
    echo "Usage: $0 -user <user> -original_device_index <idx> -devices_list <dev0-dev1> -trace <trace_path> -output_dir <out_dir> -duration <seconds>"
    exit 1
fi

output_file="${output_dir}/$(basename "${trace}")"
trace_dir=$(dirname "${trace}")
echo "user: ${user}"
echo "original_device_index: ${original_device_index}"
echo "trace_dir: ${trace_dir}"
echo "trace: ${trace}"
echo "devices_list: ${devices_list}"
echo "output_dir: ${output_dir}"
echo "output_file: ${output_file}"
echo "duration: ${duration}"

mkdir -p "${output_dir}/"

generate_stats_path() {
    local trace_name
    trace_name=$(basename "${trace}")
    echo "${output_dir}/${trace_name}.stats"
}

generate_cpu_overhead_path() {
    local trace_name
    trace_name=$(basename "${trace}")
    echo "${output_dir}/${trace_name}.csv"
}

replay_file() {
    local stats_path
    local cpu_overhead_path
    stats_path=$(generate_stats_path)
    cpu_overhead_path=$(generate_cpu_overhead_path)

    # CPU usage before
    grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> "${cpu_overhead_path}"

    # Run surrogate DT-based replay
    sudo ./io_replayer_dt "${original_device_index}" "${devices_list}" "${trace}" "${output_file}" "${duration}"

    # CPU usage after
    grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}' >> "${cpu_overhead_path}"

    echo "output replayed trace : ${output_file}"
    echo "         output stats : ${stats_path}"

    if [ -n "${user}" ]; then
        chown -R "${user}:${user}" "${trace_dir}" || true
    fi
}

replay_file

