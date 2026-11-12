#!/usr/bin/env python3

# Copyright (C) 2026 Tamarind Bio
#
# This file is part of AdaptiveFlow.
#
# AdaptiveFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# AdaptiveFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AdaptiveFlow.  If not, see <https://www.gnu.org/licenses/>.

"""The AWS Batch resourceRequirements of a subjob, for every place that submits one.

VCPU and MEMORY as always. A job that docks on a GPU (a GPU program in
afvs_run.py: Uni-Dock) also sets aws_batch_subjob_gpus in all.ctrl, and gets a
GPU requirement; without it the list is exactly what it was before GPUs.
"""

import re


def gpu_count(config):
    """aws_batch_subjob_gpus as an int: absent or empty is 0."""
    value = config.get('aws_batch_subjob_gpus', "")
    if value == "":
        return 0
    if not re.fullmatch(r'[0-9]+', str(value)):
        raise ValueError("'aws_batch_subjob_gpus' must be a whole number")
    return int(value)


def resource_requirements(config):
    requirements = [
        {'type': 'VCPU', 'value': config['aws_batch_subjob_vcpus']},
        {'type': 'MEMORY', 'value': config['aws_batch_subjob_memory']},
    ]
    gpus = gpu_count(config)
    if gpus:
        requirements.append({'type': 'GPU', 'value': str(gpus)})
    return requirements
