#!/bin/bash

# Reset script for Astraeus document database
# This script deletes all documents and resets the vector database

# Set the script directory to the project root
cd "$(dirname "$0")" || { echo "Error: Could not change to script directory"; exit 1; }

echo "======================================================="
echo "Astraeus Database Reset Tool"
echo "======================================================="
echo "This will delete ALL uploaded documents and reset the vector database."
echo "WARNING: This operation cannot be undone!"
echo ""

# Ask for confirmation
read -p "Are you sure you want to proceed? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Make the reset_database.py script executable
chmod +x reset_database.py

# Run the reset script
echo ""
echo "Starting database reset..."
python3 reset_database.py

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================="
    echo "Database reset completed successfully."
    echo "======================================================="
else
    echo ""
    echo "======================================================="
    echo "Error: Failed to reset database."
    echo "======================================================="
    exit 1
fi 