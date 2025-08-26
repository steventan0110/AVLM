#!/bin/bash

# This file contains global variables and functions for the project.
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set the root directory of the project (parent directory of scripts)
export ROOT="$(dirname "$SCRIPT_DIR")"
export REPO="$ROOT"

# Set data paths relative to ROOT or use environment variables if available
export IEMOCAP="${IEMOCAP_PATH:-$ROOT/dataset/IEMOCAP/processed}"
export RECOLA="${RECOLA_PATH:-$ROOT/dataset/RECOLA_public/processed_evenly_chunk}"
export LRS="${LRS_PATH:-$ROOT/dataset/MUAVIC/muavic/en}"
export LRSNEW="${LRSNEW_PATH:-$ROOT/dataset/MUAVIC/muavic/en/audio_clean}"
export LRS3DMMSMALL="${LRS3DMMSMALL_PATH:-$ROOT/dataset/MUAVIC/muavic/en/40hr}"
export LRS3SMIRK="${LRS3SMIRK_PATH:-$ROOT/dataset/MUAVIC/smirk/filter}"
export LRS3SMIRK_LARGE="${LRS3SMIRK_LARGE_PATH:-$ROOT/dataset/MUAVIC/smirk/filter_large}"
export EMOWRITE="${EMOWRITE_PATH:-$ROOT/dataset/IEMOCAP/conv_rewrite/processed}"

# Set checkpoint directory relative to ROOT or use environment variable if available
export SPIRITLM_CHECKPOINTS_DIR="${SPIRITLM_CHECKPOINTS_DIR:-$ROOT/checkpoints/spiritlm}"