#!/bin/bash

# Simple Shell Script to Clone a Fixed Public GitHub Repository

# Fixed GitHub repository URL
REPO_URL="https://github.com/Dheeraj070/RAG-chatbot.git"

# Extract the repository name from the URL
REPO_NAME=$(basename "$REPO_URL" .git)

# Check if the directory already exists
if [ -d "$REPO_NAME" ]; then
    echo "⚠️ Directory '$REPO_NAME' already exists. Please remove it or use a different URL."
    exit 1
fi

# Clone the repository
echo "⬇️ Cloning repository from $REPO_URL ..."
git clone "$REPO_URL"

# Check if the clone was successful
if [ $? -eq 0 ]; then
    echo "✅ Repository cloned successfully into '$REPO_NAME'"
else
    echo "❌ Failed to clone repository."
    exit 1
fi
