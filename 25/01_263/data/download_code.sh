#!/bin/bash

# Directory to store downloaded repos
mkdir -p downloaded_repos

# List of public GitHub repo URLs
REPOS=(
  "https://github.com/harshithbn63/tree/main/25/01_263/Code"
)

# Clone each repo into downloaded_repos/
for REPO in "${REPOS[@]}"; do
  git clone $REPO downloaded_repos/$(basename $REPO)
done

echo "Download complete."
