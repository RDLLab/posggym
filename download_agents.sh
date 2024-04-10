#!/bin/bash

# Downloads all agent models on Linux/macOS
agent_model_repo_url=https://github.com/RDLLab/posggym-agent-models/archive/refs/tags/v0.6.0.tar.gz

function download_agent_models() {
	echo -e "\nDownloading agent models..."
	echo -e "This may take a few minutes."
	script_path=$(dirname "${BASH_SOURCE[0]}")
	asset_dir=$script_path/posggym/assets
	mkdir -p $asset_dir
	curl -L $agent_model_repo_url \
		| tar -xz --strip=2 --directory=$asset_dir

	echo -e "Done downloading agent models!"
}

function main() {
  download_agent_models
}

main "$@"
