# PowerShell script to download a public Google Drive folder using gdown
pip install gdown
# === Set the Google Drive folder URL ===
$folderUrl = "https://drive.google.com/drive/folders/1GWUb5jgzqGG9PVbvRa4nQv-dsCs6MgqQ?usp=drive_link"

# === Set the destination directory ===
$destinationDir = ".\downloaded_files"

# === Create the destination directory if it doesn't exist ===
if (-not (Test-Path -Path $destinationDir)) {
    New-Item -ItemType Directory -Path $destinationDir | Out-Null
}

# === Run gdown to download the folder ===
Write-Host "Downloading from $folderUrl into $destinationDir..."
python -m gdown --folder $folderUrl -O $destinationDir

Write-Host "`n✅ Download completed! Files are saved in $destinationDir"