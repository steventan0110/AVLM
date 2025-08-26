# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

from setuptools import find_packages, setup

NAME = "spiritlm"
VERSION = "0.1.0"
DESCRIPTION = "Interleaved Spoken and Written Language Model"
URL = "https://github.com/facebookresearch/spiritlm"
KEYWORDS = [
    "Language Model, Speech Language Model, Multimodal, Crossmodal, Expressivity Modeling"
]
LICENSE = "FAIR Noncommercial Research License"


def _get_long_description():
    with (Path(__file__).parent / "README.md").open(encoding="utf-8") as file:
        long_description = file.read()
    return long_description


def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [
            s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))
        ]


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=_get_long_description(),
    long_description_content_type="text/plain",
    url=URL,
    license=LICENSE,
    author="Meta",
    keywords=KEYWORDS,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: FAIR Noncommercial Research License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=_read_reqs("requirements.txt"),
    extras_require={
        "dev": ["pytest"],
        "eval": ["pandas"],
    },
)
