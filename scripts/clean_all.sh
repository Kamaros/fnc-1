#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$DIR" > /dev/null
./clean_features.sh && ./clean_models.sh && ./clean_preprocessing.sh
popd > /dev/null