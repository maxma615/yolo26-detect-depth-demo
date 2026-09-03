# Packaging (.deb via CPack)

The CMake project ships CPack config so you can produce an installable `.deb`
for the RDK S100P (arm64).

## Build the .deb (on the board, or an aarch64 env)

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd build
cpack -G DEB
# -> yolo26-detect-depth-demo_1.0.0_arm64.deb
```

## Install

```bash
sudo dpkg -i yolo26-detect-depth-demo_1.0.0_arm64.deb
```

Installs to `/opt/yolo26_dual/` (binary, `web/`, `assets/`, `run.sh`).

## Notes

- **Models are not packaged.** They are ~200MB and fetched separately via
  `scripts/download_models.sh`. Point the installed `run.sh` at your model dir.
- `CPACK_DEBIAN_PACKAGE_DEPENDS` lists `libopencv-dev, libdrm2`; the Horizon
  runtime (`libdnn.so`, `libhbucp.so`) is part of the RDK image and is referenced
  via RPATH `/usr/hobot/lib`.
- For a kiosk/autostart install, copy `scripts/yolo26-dual.service` to
  `/etc/systemd/system/` and adjust `ExecStart` to the installed path, or use
  `scripts/install_boot.sh` from a source checkout.

## Alternative: source + deploy script

If you prefer not to use a package:

```bash
BOARD=root@<ip> ./scripts/deploy.sh   # rsync + on-board build
```
