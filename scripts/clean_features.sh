#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$DIR" > /dev/null
find ../caches/features ! -name '.gitignore' -type f -exec rm -f {} +
popd > /dev/null