#!/bin/bash

# Update package list
echo "Updating package list..."
sudo apt update

# Install necessary packages
echo "Installing dependencies..."
sudo apt install -y libgl1 \
                    libx11-6 \
                    libx11-dev \
                    libgl1-mesa-glx \
                    libegl1 \
                    libegl1-mesa \
                    libgles2-mesa

echo "Installation complete!"