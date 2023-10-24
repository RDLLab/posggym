#!/bin/bash

# Downloads all agent models on Linux/macOS

function download_agent_models() {
	echo -e "\nDownloading agent models..."
	echo -e "This may take a few minutes."
	script_path=$(dirname "${BASH_SOURCE[0]}")
	asset_dir=$script_path/posggym/assets
	mkdir -p $asset_dir
	curl -L https://github.com/RDLLab/posggym-agent-models/tarball/main \
		| tar -xz --strip=2 --directory=$asset_dir

	echo -e "Done downloading agent models!"
}

function main() {
  download_agent_models
}

main "$@"
