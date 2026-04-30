#!/bin/bash

# Script to recursively delete all :Zone.Identifier files
# These files are typically Windows/WSL metadata and are often unwanted on Linux.

TARGET_DIR=${1:-"."}

echo "Searching for :Zone.Identifier files in $TARGET_DIR..."

# Count files before deletion
COUNT=$(find "$TARGET_DIR" -name "*:Zone.Identifier" | wc -l)

if [ "$COUNT" -eq 0 ]; then
    echo "No :Zone.Identifier files found."
    exit 0
fi

echo "Found $COUNT files. Deleting..."

# Delete the files
find "$TARGET_DIR" -name "*:Zone.Identifier" -delete

echo "Cleanup complete."
