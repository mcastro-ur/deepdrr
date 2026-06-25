#!/usr/bin/env bash
# setup_ubuntu.sh — installs apt dependencies and configures the NVIDIA EGL driver.
# Run with: sudo ./scripts/setup_ubuntu.sh
#
# WSL2 note: in a WSL2 environment the NVIDIA kernel driver is supplied by the
# Windows host; this script skips native driver installation and instead points
# the EGL vendor config to the WSL2 NVIDIA library location automatically.
set -e

# Detect WSL2 environment
IS_WSL2=false
if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
    IS_WSL2=true
    echo "WSL2 environment detected — using WSL2 NVIDIA library path for EGL."
fi

# Configure NVIDIA EGL vendor
mkdir -p /usr/share/glvnd/egl_vendor.d/

if [ "$IS_WSL2" = true ]; then
    # In WSL2, the NVIDIA userspace libraries are provided by the Windows host
    # driver and are available at /usr/lib/wsl/lib/.
    EGL_LIB_PATH="/usr/lib/wsl/lib/libEGL_nvidia.so.0"
else
    EGL_LIB_PATH="libEGL_nvidia.so.0"
fi

cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json << EOF
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "${EGL_LIB_PATH}"
    }
}
EOF

apt-get update

# Should be the same list of packages as in the Dockerfile, make sure to keep them in sync
apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    freeglut3-dev
