// KMS HDMI output for the YOLO26 dual demo (RDK S100P).
// Owns /dev/dri/card0 as DRM master, modesets the first connected
// connector (prefers HDMI-A-1), and presents the composite frame with
// double-buffered page flips. Falls back to a polling re-init when no
// display is attached (hotplug pickup within ~2 s).
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <xf86drm.h>
#include <xf86drmMode.h>

#include "opencv2/opencv.hpp"

class KmsDisplay {
 public:
  ~KmsDisplay() { Close(); }

  // Try to acquire a connected connector + crtc and modeset it.
  // Returns false (and stays inert) when no monitor is attached.
  bool TryInit(const char* device = "/dev/dri/card0") {
    Close();
    fd_ = open(device, O_RDWR | O_CLOEXEC);
    if (fd_ < 0) return false;
    if (!Setup()) {
      Close();
      return false;
    }
    std::printf("[kms  ] %s: %dx%d@%d on %s (fb %dx%d)\n", device, mode_.hdisplay,
                mode_.vdisplay, mode_.vrefresh, connector_name_.c_str(), fb_w_,
                fb_h_);
    return true;
  }

  bool Ready() const { return ready_; }
  int Width() const { return mode_.hdisplay; }
  int Height() const { return mode_.vdisplay; }

  // Present one BGR composite: letterboxed into the scanout buffer and
  // page-flipped at vblank. Blocks until the previous flip completed.
  void Present(const cv::Mat& bgr) {
    if (!ready_ || bgr.empty()) return;
    // letterbox geometry (constant per resolution pair)
    const double s = std::min((double)fb_w_ / bgr.cols,
                              (double)fb_h_ / bgr.rows);
    const int dw = std::max(1, (int)(bgr.cols * s + 0.5));
    const int dh = std::max(1, (int)(bgr.rows * s + 0.5));
    const int dx = (fb_w_ - dw) / 2, dy = (fb_h_ - dh) / 2;
    cv::Mat dst(dh, dw, CV_8UC4, (uint8_t*)back_->map + dy * back_->stride +
                                     dx * 4, back_->stride);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(dw, dh), 0, 0, cv::INTER_AREA);
    cv::cvtColor(resized, dst, cv::COLOR_BGR2BGRA);  // scanout is XBGR8888
    // page flip (wait for the previous flip's vblank first)
    const uint32_t new_fb = (back_ == bufs_.data()) ? bufs_[1].fb : bufs_[0].fb;
    if (drmModePageFlip(fd_, crtc_id_, new_fb, DRM_MODE_PAGE_FLIP_EVENT,
                        nullptr) != 0) {
      // e.g. monitor unplugged mid-run — re-enter discovery
      std::fprintf(stderr, "[kms  ] pageflip failed, reinitializing\n");
      ready_ = false;
      return;
    }
    back_ = (back_ == bufs_.data()) ? bufs_.data() + 1 : bufs_.data();
    WaitForFlipEvent();
    flips_++;
  }

  uint64_t flips() const { return flips_; }

  void Close() {
    ready_ = false;
    for (auto& b : bufs_) {
      if (b.map && b.map != MAP_FAILED) munmap(b.map, b.size);
      if (b.fb) drmModeRmFB(fd_, b.fb);
      if (b.handle) {
        drm_mode_destroy_dumb d{};
        d.handle = b.handle;
        drmIoctl(fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
      }
    }
    bufs_.clear();
    if (res_) drmModeFreeResources(res_);
    if (fd_ >= 0) {
      drmModeSetCrtc(fd_, crtc_id_, 0, 0, 0, nullptr, 0, nullptr);
      close(fd_);
    }
    fd_ = -1;
    res_ = nullptr;
    crtc_id_ = 0;
  }

 private:
  struct Buf {
    uint32_t handle = 0, fb = 0;
    uint32_t stride = 0;
    uint64_t size = 0;
    void* map = nullptr;
  };

  bool Setup() {
    res_ = drmModeGetResources(fd_);
    if (!res_) return false;
    drmModeConnectorPtr best = nullptr;
    // prefer HDMI, else any connected connector
    for (int i = 0; i < res_->count_connectors; ++i) {
      drmModeConnectorPtr c = drmModeGetConnector(fd_, res_->connectors[i]);
      if (!c) continue;
      const std::string name = ConnectorName(c);
      if (c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
        if (name.find("HDMI") != std::string::npos) {
          best = c;
          connector_name_ = name;
          connector_id_ = c->connector_id;
          break;
        }
        if (!best) {
          best = c;
          connector_name_ = name;
          connector_id_ = c->connector_id;
        }
      }
      drmModeFreeConnector(c);
    }
    if (!best) return false;

    // preferred mode, else the first (highest) one
    drmModeModeInfo mode = best->modes[0];
    for (int m = 0; m < best->count_modes; ++m)
      if (best->modes[m].type & DRM_MODE_TYPE_PREFERRED) {
        mode = best->modes[m];
        break;
      }
    mode_ = mode;

    // encoder -> crtc
    drmModeEncoderPtr enc = nullptr;
    if (best->encoder_id) enc = drmModeGetEncoder(fd_, best->encoder_id);
    if (enc && enc->crtc_id) {
      crtc_id_ = enc->crtc_id;
    } else if (enc) {
      crtc_id_ = res_->crtcs[0];
    } else {
      crtc_id_ = res_->crtcs[0];
    }
    if (enc) drmModeFreeEncoder(enc);
    drmModeFreeConnector(best);

    // double-buffered dumb framebuffers at the screen resolution
    fb_w_ = mode_.hdisplay;
    fb_h_ = mode_.vdisplay;
    bufs_.resize(2);
    for (auto& b : bufs_) {
      drm_mode_create_dumb create{};
      create.width = fb_w_;
      create.height = fb_h_;
      create.bpp = 32;
      if (drmIoctl(fd_, DRM_IOCTL_MODE_CREATE_DUMB, &create) != 0) return false;
      b.handle = create.handle;
      b.stride = create.pitch;
      b.size = create.size;
      drm_mode_map_dumb m{};
      m.handle = create.handle;
      if (drmIoctl(fd_, DRM_IOCTL_MODE_MAP_DUMB, &m) != 0) return false;
      b.map = mmap(nullptr, b.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                   m.offset);
      if (b.map == MAP_FAILED) return false;
      std::memset(b.map, 0, b.size);  // black borders
      if (drmModeAddFB(fd_, fb_w_, fb_h_, 24, 32, b.stride, b.handle, &b.fb) !=
          0)
        return false;
    }
    back_ = bufs_.data();

    if (drmSetMaster(fd_) != 0)
      std::fprintf(stderr, "[kms  ] drmSetMaster failed (display in use?)\n");

    if (drmModeSetCrtc(fd_, crtc_id_, bufs_[0].fb, 0, 0, &connector_id_, 1,
                       &mode_) != 0)
      return false;
    ready_ = true;
    return true;
  }

  void WaitForFlipEvent() {
    drmEventContext ev{};
    ev.version = 2;
    ev.vblank_handler = [](int, unsigned int, unsigned int, unsigned int,
                           void*) {};
    // blocking read of the flip completion event = vblank sync
    drmHandleEvent(fd_, &ev);
  }

  std::string ConnectorName(drmModeConnectorPtr c) const {
    std::string n = drmModeGetConnectorTypeName(c->connector_type);
    return n + "-" + std::to_string(c->connector_type_id);
  }

  int fd_ = -1;
  drmModeResPtr res_ = nullptr;
  uint32_t crtc_id_ = 0, connector_id_ = 0;
  drmModeModeInfo mode_{};
  std::string connector_name_;
  std::vector<Buf> bufs_;
  Buf* back_ = nullptr;
  int fb_w_ = 0, fb_h_ = 0;
  bool ready_ = false;
  uint64_t flips_ = 0;
};
