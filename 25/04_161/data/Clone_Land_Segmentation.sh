#!/bin/bash

# URL of the public GitHub repository
REPO_URL="https://github.com/SaiKeertisahukari/LandSegmentation.git"

# Clone the repository
git clone "$REPO_URL"

# Check if cloning was successful
if [ $? -eq 0 ]; then
    echo "Repository cloned successfully"
else
    echo "Failed to clone repository"
    exit 1
fi
