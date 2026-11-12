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

# Build the sealed CPU and GPU images (tools/docker/Dockerfile.sealed) from this checkout, and
# optionally push them to an ECR repository:
#
#   tools/docker/build_sealed.sh                                   build both, tagged locally
#   tools/docker/build_sealed.sh --push ACCOUNT.dkr.ecr.REGION.amazonaws.com/vf-afvs-ecr
#
# The tags are sealed-cpu-<sha> and sealed-gpu-<sha>, <sha> the first 7 characters of this
# checkout's commit, so the checkout must be clean: the tag has to name what is in the image. A
# tag already in the repository is never overwritten (the repository's tags are mutable). The
# image labels carry the full commit. Uses the AWS credentials and profile of the environment.

set -euo pipefail

push=""
if [ "${1:-}" = "--push" ]; then
    push="${2:?--push needs the repository URI}"
elif [ $# -ne 0 ]; then
    echo "usage: $0 [--push REPOSITORY_URI]" >&2
    exit 2
fi

cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain)" ]; then
    echo "The working tree is not clean; commit or remove the changes, the image tag names a commit." >&2
    exit 1
fi
full_sha="$(git rev-parse HEAD)"
sha="$(git rev-parse --short=7 HEAD)"

repository="${push:-afvs-sealed}"
tags=()
for flavor in cpu gpu; do
    tag="sealed-${flavor}-${sha}"
    if [ -n "$push" ]; then
        region="$(echo "$push" | sed -E 's/^[0-9]+\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com\/.*/\1/')"
        name="${push#*/}"
        if aws ecr describe-images --region "$region" --repository-name "$name" --image-ids "imageTag=${tag}" > /dev/null 2>&1; then
            echo "${repository}:${tag} already exists; not overwriting it." >&2
            exit 1
        fi
    fi
    tags+=("$tag")
done

for flavor in cpu gpu; do
    docker build --platform linux/amd64 --target "$flavor" \
        --build-arg "AFVS_REVISION=${full_sha}" \
        -t "${repository}:sealed-${flavor}-${sha}" \
        -f tools/docker/Dockerfile.sealed .
done

if [ -n "$push" ]; then
    aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "${push%%/*}"
    for tag in "${tags[@]}"; do
        docker push "${repository}:${tag}"
    done
fi

for tag in "${tags[@]}"; do
    docker image inspect --format '{{.Id}} {{index .Config.Labels "org.opencontainers.image.revision"}}' "${repository}:${tag}" | sed "s|^|${repository}:${tag} |"
    if [ -n "$push" ]; then
        docker image inspect --format '{{index .RepoDigests 0}}' "${repository}:${tag}"
    fi
done
