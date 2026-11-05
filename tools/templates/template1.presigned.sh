#!/usr/bin/env bash

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

# ---------------------------------------------------------------------------
#
# Description: AWS Batch job command for job_storage_mode=presigned (the
# sealed image, which has no ENTRYPOINT). The job definition's command is
#
#   ["/opt/AFVS/tools/templates/template1.presigned.sh"]
#
# ---------------------------------------------------------------------------

set -eu

export AFVS_WORKUNIT_SUBJOB=${AWS_BATCH_JOB_ARRAY_INDEX:?every presigned job is a Batch array child}

cd "${AFVS_HOME:-/opt/AFVS}/tools/templates/"
exec python3 ./afvs_run.py
