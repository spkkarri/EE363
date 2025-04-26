#!/bin/bash

# URL of the public Google Drive folder
FOLDER_URL="https://drive.google.com/drive/folders/1cSSW37KylrJBFe8SyvbA3dULMy-b8Fib?usp=sharing"

# Optional: Specify the download destination
DEST_DIR="./downloaded_files"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Use gdown to download the entire folder
echo "Downloading from $FOLDER_URL into $DEST_DIR..."
gdown --folder "$FOLDER_URL" -O "$DEST_DIR"

echo "✅ Download completed! Files are saved in $DEST_DIR"