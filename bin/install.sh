#!/usr/bin/env bash
# Create the conda environment and download the ARQMath Task 1 files.
#
# The environment is named cos487-final. Activate it with:
#   conda activate cos487-final
#
# Collection XML lands in data/. Topics and the official test qrels land in
# data/topics and data/qrels, under the names the readers expect. Files that
# are already present are left alone. Posts.V1.3.xml is about 4 GB.
#
# Source folder:
# https://drive.google.com/drive/folders/1ZPKIWDnhMGRaPNVLi1reQxZWTfH2R4u3

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
data="$root/data"
env_name="cos487-final"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or Anaconda, then rerun bin/install.sh." >&2
  exit 1
fi

set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u

if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
  conda create -n "$env_name" python=3.12 -y
fi

set +u
conda activate "$env_name"
set -u
python -m pip install -r "$root/requirements.txt"
gdown=(gdown)

download() {
  local id="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -s "$dest" ]]; then
    echo "skip  $dest"
    return
  fi
  echo "get   $dest"
  "${gdown[@]}" --continue --retries 3 "https://drive.google.com/uc?id=${id}" -O "$dest"
}

# Collection, version V1.3. These names match DataReaderRecord.
download 1bFrbVf2WFFAKfTrv3L2DgZWuk6DQ_oL2 "$data/Badges.V1.3.xml"
download 1vDn_8sCHm_6lg4JuJBxlVsUEPU5KL4Mv "$data/Comments.V1.3.xml"
download 1AjZ2SDH7XSG5xKwfqpXwwvJkUuQuL0pr "$data/PostLinks.V1.3.xml"
download 14SSwTqLZgLVP6iDsAJbmxgb01a8NYyDb "$data/Posts.V1.3.xml"
download 1fcLhlJqGaXm3J2_b4yar7mLe8Oau1kbQ "$data/Tags.V1.3.xml"
download 1bUxHKTu9zDwq1TFPDRU213pDDN40RKX1 "$data/Users.V1.3.xml"
download 1jrDxE_NtIrZXh7o6OuJNQg233YrpQga8 "$data/Votes.V1.3.xml"

# Task 1 topics. Drive names are the versioned originals.
download 1iFGWmQkPDPzPMoSLGrRt_VgfuPWYBc-e "$data/topics/Topics_Task1_2020.xml"
download 1IELdhLzrx_6nD47e_8gbxFM6QkEOsyjz "$data/topics/Topics_Task1_2021.xml"
download 1EimgCZPeg1mBuQcovHCG-GAegzs7cZM4 "$data/topics/Topics_Task1_2022.xml"

# Official Task 1 test judgments, not the later "all" files.
download 1qUtdQbgqmIJDmMQoae7UQ7fe999B87om "$data/qrels/task1_2020.tsv"
download 1fEoHkHIxEpsvHCzMOjcgM5Q3se8MkVeB "$data/qrels/task1_2021.tsv"
download 1loBQi_Zkamw5JHlkAlfptq0loZ-rsjOP "$data/qrels/task1_2022.tsv"
