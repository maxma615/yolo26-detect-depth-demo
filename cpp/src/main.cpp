// ============================================================================
// YOLO26 Detect + YOLO26 Depth dual-model concurrent inference demo (C++)
// Target: RDK S100P (Nash, march nash-m), TROS/Hobot stack.
//
// Inference: hobot_dnn C API (hbDNN* / hbUCP*, libdnn.so) — the same native
// stack TROS components use. Each model owns a worker thread that submits an
// async UCP task (priority + BPU core mask) so both models are resident on
// the single-core BPU scheduler at the same time ("simultaneous inference"),
// while each thread's CPU-side pre/post processing overlaps the other
// model's BPU execution.
//
// Pipeline: V4L2(MJPG) capture -> FrameHub(latest-wins) -> {detect,depth}
// workers -> Compositor (top: frame+boxes, bottom: depth turbo colormap)
// -> MJPEG HTTP server + stats.json side panel.
// ============================================================================

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <csignal>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "opencv2/opencv.hpp"
#include "kms_display.h"
#include "system_metrics.h"
#include "tracking/byte_tracker.h"
#include "ui/browser_overlay.h"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

static std::atomic<bool> g_stop{false};

#define NOW() (std::chrono::steady_clock::now())
using fclock = std::chrono::steady_clock;
static inline double ms_since(fclock::time_point t) {
  return std::chrono::duration<double, std::milli>(fclock::now() - t).count();
}

#define HB_CHECK(expr, msg)                                                  \
  do {                                                                       \
    int32_t _rc = (expr);                                                    \
    if (_rc != 0) {                                                          \
      std::fprintf(stderr, "[FATAL] %s failed rc=%d (%s:%d)\n", #expr, _rc,  \
                   __FILE__, __LINE__);                                      \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

static std::string read_first_line(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return "";
  char buf[256] = {0};
  if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0;
  std::fclose(f);
  std::string s(buf);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}

static std::string basename_of(const std::string& p) {
  size_t pos = p.find_last_of('/');
  return pos == std::string::npos ? p : p.substr(pos + 1);
}

// ---------------------------------------------------------------------------
// hbDNN model wrapper
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// hbDNN model wrapper
// ---------------------------------------------------------------------------

static inline int64_t Align64(int64_t x) { return (x + 63) & ~63LL; }

// Normalize a tensor's layout exactly like hbm_runtime does:
// fill dynamic (-1) strides from innermost out with 64-byte alignment, and
// derive the memory size from stride[0] * dim[0] when alignedByteSize is -1.
// hbDNNInferV2 REJECTS tensors whose stride array still contains -1.
static void NormalizeTensorLayout(hbDNNTensorProperties* p) {
  auto& s = p->stride;
  const auto& d = p->validShape;
  const int n = d.numDimensions;
  for (int idx = n - 1; idx >= 0; --idx) {
    if (s[idx] < 0) {
      if (idx == n - 1) {
        std::fprintf(stderr, "[FATAL] last-dim stride is dynamic\n");
        std::exit(1);
      }
      s[idx] = (int32_t)Align64((int64_t)s[idx + 1] *
                                d.dimensionSize[idx + 1]);
    }
  }
  if (p->alignedByteSize < 0)
    p->alignedByteSize = (int64_t)s[0] * d.dimensionSize[0];
}

class DnnModel {
 public:
  DnnModel(hbDNNPackedHandle_t packed, const std::string& model_name) {
    name_ = model_name;
    HB_CHECK(hbDNNGetModelHandle(&handle_, packed, model_name.c_str()),
             "hbDNNGetModelHandle");
    HB_CHECK(hbDNNGetInputCount(&input_count_, handle_), "GetInputCount");
    HB_CHECK(hbDNNGetOutputCount(&output_count_, handle_), "GetOutputCount");
    inputs_.resize(input_count_);
    outputs_.resize(output_count_);
    for (int32_t i = 0; i < input_count_; ++i) {
      const char* n = nullptr;
      hbDNNGetInputName(&n, handle_, i);
      if (n) input_names_.emplace_back(n);
      HB_CHECK(
          hbDNNGetInputTensorProperties(&inputs_[i].properties, handle_, i),
          "GetInputTensorProperties");
      NormalizeTensorLayout(&inputs_[i].properties);
      HB_CHECK(hbUCPMallocCached(
                   &inputs_[i].sysMem,
                   (uint64_t)inputs_[i].properties.alignedByteSize, 0),
               "hbUCPMallocCached(input)");
    }
    for (int32_t i = 0; i < output_count_; ++i) {
      const char* n = nullptr;
      hbDNNGetOutputName(&n, handle_, i);
      if (n) output_names_.emplace_back(n);
      HB_CHECK(
          hbDNNGetOutputTensorProperties(&outputs_[i].properties, handle_, i),
          "GetOutputTensorProperties");
      NormalizeTensorLayout(&outputs_[i].properties);
      HB_CHECK(hbUCPMallocCached(
                   &outputs_[i].sysMem,
                   (uint64_t)outputs_[i].properties.alignedByteSize, 0),
               "hbUCPMallocCached(output)");
    }
  }

  ~DnnModel() {
    for (auto& t : inputs_)
      if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
    for (auto& t : outputs_)
      if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
    // hbDNNGetModelHandle returns a model borrowed from the packed handle.
    // hbDNNRelease accepts only that owning packed handle, which main releases
    // after all worker threads have stopped.
  }

  // Submit an async task and wait for completion.
  // Inputs must already be written into input memory.
  // Returns latency in ms (submit -> done, includes BPU queueing).
  double Run(int32_t priority, uint64_t backend) {
    for (auto& t : inputs_)
      hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_CLEAN);
    hbUCPTaskHandle_t task = nullptr;
    HB_CHECK(hbDNNInferV2(&task, outputs_.data(), inputs_.data(), handle_),
             "hbDNNInferV2");
    hbUCPSchedParam sp{};
    sp.priority = priority;
    sp.customId = 0;
    sp.backend = backend;
    sp.deviceId = 0;
    HB_CHECK(hbUCPSubmitTask(task, &sp), "hbUCPSubmitTask");
    auto t0 = NOW();
    HB_CHECK(hbUCPWaitTaskDone(task, 5000), "hbUCPWaitTaskDone");
    double ms = ms_since(t0);
    for (auto& t : outputs_)
      hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    HB_CHECK(hbUCPReleaseTask(task), "hbUCPReleaseTask");
    return ms;
  }

  // Input i as a writable cv::Mat wrapping the tensor memory (U8, HxW or
  // HxWx2). Caller must ensure shape matches the model contract.
  cv::Mat InputAsMat(int i, int rows, int cols, int type) {
    return cv::Mat(rows, cols, type, inputs_[i].sysMem.virAddr);
  }
  void* InputMem(int i) { return inputs_[i].sysMem.virAddr; }
  void* OutputMem(int i) { return outputs_[i].sysMem.virAddr; }
  const hbDNNTensorProperties& InputProps(int i) const {
    return inputs_[i].properties;
  }
  const hbDNNTensorProperties& OutputProps(int i) const {
    return outputs_[i].properties;
  }
  int32_t input_count() const { return input_count_; }
  int32_t output_count() const { return output_count_; }
  const std::vector<std::string>& input_names() const { return input_names_; }
  const std::vector<std::string>& output_names() const {
    return output_names_;
  }
  const std::string& name() const { return name_; }

 private:
  std::string name_;
  hbDNNHandle_t handle_ = nullptr;
  int32_t input_count_ = 0, output_count_ = 0;
  std::vector<hbDNNTensor> inputs_, outputs_;
  std::vector<std::string> input_names_, output_names_;
};

// BGR -> NV12 planes written directly into model input tensors.
// y_mat: CV_8UC1 (h, w); uv_mat: CV_8UC2 (h/2, w/2).
static void BgrToNv12Inputs(const cv::Mat& bgr, cv::Mat y_mat,
                            cv::Mat uv_mat) {
  const int h = bgr.rows, w = bgr.cols;
  cv::Mat i420(h * 3 / 2, w, CV_8UC1);
  cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
  uint8_t* base = i420.data;
  std::memcpy(y_mat.data, base, (size_t)h * w);
  const uint8_t* u = base + (size_t)h * w;
  const uint8_t* v = u + (size_t)(h / 2) * (w / 2);
  uint8_t* uv = uv_mat.data;
  const size_t n = (size_t)(h / 2) * (w / 2);
  for (size_t i = 0; i < n; ++i) {
    uv[2 * i] = u[i];
    uv[2 * i + 1] = v[i];
  }
}

// ---------------------------------------------------------------------------
// Detection (YOLO26 anchor-free, LTRB decoding) — mirrors rdk_model_zoo
// utils/py_utils/postprocess.py exactly.
// ---------------------------------------------------------------------------

struct Det {
  float x1, y1, x2, y2, score;
  int cls;
};

static inline float Sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

static void DecodeYolo26(const std::vector<const float*>& cls_outs,
                         const std::vector<const float*>& box_outs,
                         const std::vector<int>& feats, int num_classes,
                         float score_thres, std::vector<Det>& out) {
  const float conf_raw = -std::log(1.0f / score_thres - 1.0f);
  const int strides[3] = {8, 16, 32};
  out.clear();
  for (size_t s = 0; s < feats.size(); ++s) {
    const int fw = feats[s], fh = feats[s];
    const int n = fw * fh;
    const float* cls = cls_outs[s];
    const float* box = box_outs[s];
    for (int idx = 0; idx < n; ++idx) {
      const float* c = cls + (size_t)idx * num_classes;
      int best = 0;
      float maxl = c[0];
      for (int k = 1; k < num_classes; ++k)
        if (c[k] > maxl) {
          maxl = c[k];
          best = k;
        }
      if (maxl < conf_raw) continue;
      const float gx = (float)(idx % fw) + 0.5f;
      const float gy = (float)(idx / fw) + 0.5f;
      const float* b = box + (size_t)idx * 4;
      const float st = (float)strides[s];
      Det d;
      d.x1 = (gx - b[0]) * st;
      d.y1 = (gy - b[1]) * st;
      d.x2 = (gx + b[2]) * st;
      d.y2 = (gy + b[3]) * st;
      d.score = Sigmoid(maxl);
      d.cls = best;
      out.push_back(d);
    }
  }
}

static void ClasswiseNms(std::vector<Det>& dets, float iou_thres) {
  // bucket by class
  std::map<int, std::vector<int>> by_cls;
  for (int i = 0; i < (int)dets.size(); ++i) by_cls[dets[i].cls].push_back(i);
  std::vector<int> keep;
  for (auto& kv : by_cls) {
    auto& idx = kv.second;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
      return dets[a].score > dets[b].score;
    });
    std::vector<char> suppressed(idx.size(), 0);
    for (size_t i = 0; i < idx.size(); ++i) {
      if (suppressed[i]) continue;
      keep.push_back(idx[i]);
      const Det& a = dets[idx[i]];
      const float aw = a.x2 - a.x1, ah = a.y2 - a.y1;
      for (size_t j = i + 1; j < idx.size(); ++j) {
        if (suppressed[j]) continue;
        const Det& b = dets[idx[j]];
        const float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
        const float ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
        if (iw <= 0 || ih <= 0) continue;
        const float inter = iw * ih;
        const float iou =
            inter / (aw * ah + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-9f);
        if (iou > iou_thres) suppressed[j] = 1;
      }
    }
  }
  std::vector<Det> kept;
  kept.reserve(keep.size());
  for (int i : keep) kept.push_back(dets[i]);
  dets.swap(kept);
}

// Letterbox geometry (mirrors py_utils.preprocess.resized_image, type 1)
struct Letterbox {
  float scale;
  int pad_l, pad_t, new_w, new_h;
};
static Letterbox MakeLetterbox(int src_w, int src_h, int dst) {
  Letterbox lb;
  lb.scale = std::min((float)dst / src_h, (float)dst / src_w);
  lb.new_w = (int)(src_w * lb.scale);
  lb.new_h = (int)(src_h * lb.scale);
  int pad_w = dst - lb.new_w, pad_h = dst - lb.new_h;
  lb.pad_l = pad_w / 2;
  lb.pad_t = pad_h / 2;
  return lb;
}

// Locate the Y / UV plane inputs by shape signature, NOT by index:
// y  : 4 dims, last dim == 1 (HxWx1, full size)
// uv : 4 dims, last dim == 2 (H/2 x W/2 x 2)
// (yolo26n declares y first, yolo26x declares uv first — order varies.)
static void FindNv12Inputs(DnnModel* m, int* y_idx, int* uv_idx, int* size) {
  *y_idx = -1;
  *uv_idx = -1;
  for (int i = 0; i < m->input_count(); ++i) {
    auto& p = m->InputProps(i);
    if (p.validShape.numDimensions != 4) continue;
    if (p.validShape.dimensionSize[3] == 1)
      *y_idx = i;
    else if (p.validShape.dimensionSize[3] == 2)
      *uv_idx = i;
  }
  if (*y_idx < 0 || *uv_idx < 0) {
    std::fprintf(stderr, "[FATAL] NV12 model without y/uv inputs\n");
    std::exit(1);
  }
  *size = m->InputProps(*y_idx).validShape.dimensionSize[1];
}

class DetectModel {
 public:
  DetectModel(hbDNNPackedHandle_t packed, const std::string& name,
              const std::vector<std::string>& labels)
      : model_(packed, name), labels_(labels) {
    // Contract: inputs [y (1,S,S,1) U8, uv (1,S/2,S/2,2) U8] in either order
    FindNv12Inputs(&model_, &y_idx_, &uv_idx_, &size_);
    num_outputs_ = model_.output_count();
    // strides/feats from output shapes: [1,80,80,80],[1,80,80,4], ...
    for (int i = 0; i < num_outputs_; i += 2) {
      feats_.push_back(model_.OutputProps(i).validShape.dimensionSize[1]);
    }
  }

  double Preprocess(const cv::Mat& bgr) {
    auto t0 = NOW();
    lb_ = MakeLetterbox(bgr.cols, bgr.rows, size_);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(lb_.new_w, lb_.new_h));
    cv::copyMakeBorder(resized, letterboxed_, lb_.pad_t,
                       size_ - lb_.new_h - lb_.pad_t, lb_.pad_l,
                       size_ - lb_.new_w - lb_.pad_l, cv::BORDER_CONSTANT,
                       cv::Scalar(127, 127, 127));
    BgrToNv12Inputs(letterboxed_,
                    model_.InputAsMat(y_idx_, size_, size_, CV_8UC1),
                    model_.InputAsMat(uv_idx_, size_ / 2, size_ / 2, CV_8UC2));
    return ms_since(t0);
  }

  double Infer(int priority, uint64_t backend) {
    return model_.Run(priority, backend);
  }

  double Postprocess(int src_w, int src_h, float decode_score_thres,
                     float visible_score_thres, float nms_thres,
                     std::vector<Det>& dets,
                     std::vector<tracking::SourceDetection>& source_dets,
                     std::map<int, int>& cls_hist) {
    auto t0 = NOW();
    std::vector<const float*> cls_outs, box_outs;
    for (int i = 0; i < num_outputs_; i += 2) {
      cls_outs.push_back((const float*)model_.OutputMem(i));
      box_outs.push_back((const float*)model_.OutputMem(i + 1));
    }
    std::vector<Det> decoded;
    DecodeYolo26(cls_outs, box_outs, feats_, 80, decode_score_thres, decoded);
    ClasswiseNms(decoded, nms_thres);
    // scale back to source resolution (scale_coords_back, letterbox)
    const float scale = lb_.scale;
    const float pad_w = (size_ - src_w * scale) / 2.0f;
    const float pad_h = (size_ - src_h * scale) / 2.0f;
    dets.clear();
    source_dets.clear();
    cls_hist.clear();
    for (auto& d : decoded) {
      d.x1 = std::clamp((d.x1 - pad_w) / scale, 0.0f, (float)src_w);
      d.x2 = std::clamp((d.x2 - pad_w) / scale, 0.0f, (float)src_w);
      d.y1 = std::clamp((d.y1 - pad_h) / scale, 0.0f, (float)src_h);
      d.y2 = std::clamp((d.y2 - pad_h) / scale, 0.0f, (float)src_h);
      source_dets.push_back(
          {{d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1}, d.score, d.cls});
      if (d.score >= visible_score_thres) {
        dets.push_back(d);
        cls_hist[d.cls]++;
      }
    }
    return ms_since(t0);
  }

  int input_size() const { return size_; }
  const std::string& model_name() const { return model_.name(); }
  const std::vector<std::string>& labels() const { return labels_; }

 private:
  DnnModel model_;
  std::vector<std::string> labels_;
  int y_idx_ = -1, uv_idx_ = -1;
  int size_ = 0, num_outputs_ = 0;
  std::vector<int> feats_;
  Letterbox lb_{};
  cv::Mat letterboxed_;
};

// ---------------------------------------------------------------------------
// Depth (YOLO26 Depth, NV12 profile n/s/m and lite profile l/x)
// ---------------------------------------------------------------------------

class DepthModel {
 public:
  DepthModel(hbDNNPackedHandle_t packed, const std::string& name,
             const std::string& variant)
      : model_(packed, name), variant_(variant) {
    profile_ = (variant == "l" || variant == "x") ? "lite" : "nv12";
    if (profile_ == "lite") {
      if (variant == "x")
        cal_b_ = -0.316650390625f;
      else
        cal_b_ = -0.2498779296875f;
    }
    if (profile_ == "nv12") {
      FindNv12Inputs(&model_, &y_idx_, &uv_idx_, &size_);
    } else {
      y_idx_ = uv_idx_ = -1;
      size_ = model_.InputProps(0).validShape.dimensionSize[2];  // [1,3,S,S]
    }
    out_size_ = model_.OutputProps(0).validShape.dimensionSize[1];  // 192
  }

  double Preprocess(const cv::Mat& bgr) {
    auto t0 = NOW();
    if (profile_ == "nv12") {
      // letterbox with 114 padding (mirrors yolo26_depth.letterbox)
      const int S = size_;
      float ratio = std::min((float)S / bgr.rows, (float)S / bgr.cols);
      int rw = (int)std::round(bgr.cols * ratio);
      int rh = (int)std::round(bgr.rows * ratio);
      int padw = S - rw, padh = S - rh;
      int left = (int)std::round(padw / 2.0f - 0.1f);
      int right = padw - left;
      int top = (int)std::round(padh / 2.0f - 0.1f);
      int bottom = padh - top;
      cv::Mat resized;
      if (bgr.cols != rw || bgr.rows != rh)
        cv::resize(bgr, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
      else
        resized = bgr;
      cv::copyMakeBorder(resized, letterboxed_, top, bottom, left, right,
                         cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
      crop_ = cv::Rect(left, top, rw, rh);
      BgrToNv12Inputs(letterboxed_,
                      model_.InputAsMat(y_idx_, size_, size_, CV_8UC1),
                      model_.InputAsMat(uv_idx_, size_ / 2, size_ / 2,
                                        CV_8UC2));
    } else {
      // lite: stretched float32 RGB NCHW /255, [1,3,S,S]
      const int S = size_;
      cv::Mat rgb;
      cv::resize(bgr, rgb, cv::Size(S, S), 0, 0, cv::INTER_LINEAR);
      cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
      cv::Mat f32;
      rgb.convertTo(f32, CV_32F, 1.0 / 255.0);
      float* dst = (float*)model_.InputMem(0);
      const float* src = (const float*)f32.data;
      const size_t plane = (size_t)S * S;
      // de-interleave HWC -> CHW
      for (size_t i = 0; i < plane; ++i) {
        dst[i] = src[3 * i];
        dst[plane + i] = src[3 * i + 1];
        dst[2 * plane + i] = src[3 * i + 2];
      }
    }
    return ms_since(t0);
  }

  double Infer(int priority, uint64_t backend) {
    return model_.Run(priority, backend);
  }

  // postprocess -> colorized depth at source resolution.
  // All math (percentile + colormap) runs on the 192x192 output map, then
  // the colorized small image is geometrically restored — ~24x less CPU
  // than normalizing at source resolution, visually identical.
  double Postprocess(const cv::Mat& bgr, cv::Mat& color_out) {
    auto t0 = NOW();
    const float* raw = (const float*)model_.OutputMem(0);
    const int N = out_size_;
    cv::Mat depth(N, N, CV_32F);
    float* dptr = (float*)depth.data;
    if (profile_ == "nv12") {
      // output is calibrated log-depth already; exp at 192
      for (int i = 0; i < N * N; ++i) dptr[i] = std::exp(raw[i]);
    } else {
      for (int i = 0; i < N * N; ++i) {
        float v = std::clamp(raw[i], -4.0f, 5.0f) + cal_b_;
        dptr[i] = std::exp(v);
      }
    }
    // robust 2%/98% percentile normalization on the small map
    std::vector<float> vals(dptr, dptr + (size_t)N * N);
    float lo = 0, hi = 1;
    if (!vals.empty()) {
      const size_t n = vals.size();
      std::nth_element(vals.begin(), vals.begin() + (size_t)(0.02 * n),
                       vals.end());
      lo = vals[(size_t)(0.02 * n)];
      std::nth_element(vals.begin(), vals.begin() + (size_t)(0.98 * n),
                       vals.end());
      hi = vals[(size_t)(0.98 * n)];
    }
    cv::Mat gray(N, N, CV_8U);
    const float rng = std::max(hi - lo, 1e-6f);
    for (int i = 0; i < N * N; ++i) {
      float v = (dptr[i] - lo) / rng;
      if (v < 0) v = 0;
      if (v > 1) v = 1;
      gray.data[i] = (uint8_t)(255.0f - v * 255.0f);
    }

    // ---- distance values at grid INTERSECTIONS (0 = near) ----
    // (cols+1) x (rows+1) points; each value is the median of a 3x3 window
    // at the exact mapped pixel — an accurate point reading, not an area
    // average. The pane geometry mapping is inverted per profile
    // (lite: stretched; nv12: letterbox crop).
    if (grid_cols_ > 0 && grid_rows_ > 0) {
      const int ic = grid_cols_ + 1, ir = grid_rows_ + 1;
      grid_vals_.assign((size_t)ic * ir, 0.0f);
      for (int cy = 0; cy < ir; ++cy) {
        for (int cx = 0; cx < ic; ++cx) {
          const float u = (float)cx / grid_cols_;
          const float v = (float)cy / grid_rows_;
          float sx, sy;
          if (profile_ == "lite") {
            sx = u * N;
            sy = v * N;
          } else {
            const float xS = crop_.x + u * crop_.width;
            const float yS = crop_.y + v * crop_.height;
            sx = xS * N / (float)size_;
            sy = yS * N / (float)size_;
          }
          const int ix = std::clamp((int)std::lround(sx), 1, N - 2);
          const int iy = std::clamp((int)std::lround(sy), 1, N - 2);
          float w[9];
          int k = 0;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
              w[k++] = (float)gray.at<uchar>(iy + dy, ix + dx);
          std::nth_element(w, w + 4, w + 9);
          grid_vals_[(size_t)cy * ic + cx] = 1.0f - w[4] / 255.0f;
        }
      }
      // temporal EMA so the on-screen numbers do not flicker
      if (grid_ema_.size() == grid_vals_.size()) {
        for (size_t i = 0; i < grid_vals_.size(); ++i)
          grid_ema_[i] = 0.75f * grid_ema_[i] + 0.25f * grid_vals_[i];
      } else {
        grid_ema_ = grid_vals_;
      }
    }

    cv::Mat color_small;
    cv::applyColorMap(gray, color_small, cv::COLORMAP_TURBO);
    if (profile_ == "nv12") {
      // restore letterbox: 192 -> input square -> crop -> source size
      cv::Mat big;
      cv::resize(color_small, big, cv::Size(size_, size_), 0, 0,
                 cv::INTER_LINEAR);
      cv::Mat cropped = big(crop_);
      cv::resize(cropped, color_out, cv::Size(bgr.cols, bgr.rows), 0, 0,
                 cv::INTER_LINEAR);
    } else {
      // lite: input was stretched, no letterbox geometry
      cv::resize(color_small, color_out, cv::Size(bgr.cols, bgr.rows), 0, 0,
                 cv::INTER_LINEAR);
    }
    return ms_since(t0);
  }

  int input_size() const { return size_; }
  const std::string& profile() const { return profile_; }
  const std::string& model_name() const { return model_.name(); }
  float cal_b() const { return cal_b_; }
  // Grid of per-cell relative distances (EMA-smoothed); 0/0 disables.
  void set_grid(int cols, int rows) {
    grid_cols_ = cols;
    grid_rows_ = rows;
    grid_ema_.clear();
    grid_vals_.clear();
  }
  const std::vector<float>& grid_vals() const { return grid_ema_; }
  int grid_cols() const { return grid_cols_; }
  int grid_rows() const { return grid_rows_; }

 private:
  DnnModel model_;
  std::string variant_, profile_;
  float cal_b_ = 0.0f;
  int y_idx_ = -1, uv_idx_ = -1;
  int size_ = 0, out_size_ = 0;
  int grid_cols_ = 0, grid_rows_ = 0;
  std::vector<float> grid_vals_, grid_ema_;
  cv::Rect crop_;
  cv::Mat letterboxed_, depth_native_;
};

// ---------------------------------------------------------------------------
// Frame hub / result store
// ---------------------------------------------------------------------------

class FrameHub {
 public:
  void publish(const cv::Mat& f) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      frame_ = f;
      ++seq_;
    }
    cv_.notify_all();
  }
  // wait for a frame newer than last_seq
  bool latest(uint64_t& last_seq, cv::Mat& out, double timeout_s) {
    std::unique_lock<std::mutex> lk(mu_);
    if (seq_ <= last_seq) {
      cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [&] {
        return seq_ > last_seq || g_stop.load();
      });
    }
    if (seq_ <= last_seq || frame_.empty()) return false;
    last_seq = seq_;
    out = frame_;
    return true;
  }
  uint64_t seq() {
    std::lock_guard<std::mutex> lk(mu_);
    return seq_;
  }
  cv::Mat get() {
    std::lock_guard<std::mutex> lk(mu_);
    return frame_;
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  cv::Mat frame_;
  uint64_t seq_ = 0;
};

struct DetResult {
  std::vector<Det> dets;
  std::vector<tracking::TrackView> tracks;
  std::map<int, int> cls_hist;
  tracking::TrackerStats tracker_stats;
  double tracking_ms = 0.0;
  int object_count = 0;
  double bpu_ms = 0, pre_ms = 0, post_ms = 0;
  double stamp = 0;
  int src_w = 0, src_h = 0;
};
struct DepResult {
  cv::Mat color;
  double bpu_ms = 0, pre_ms = 0, post_ms = 0;
  double stamp = 0;
  std::vector<float> grid;
  int grid_cols = 0, grid_rows = 0;
};

class ResultStore {
 public:
  void set_det(const DetResult& r) {
    std::lock_guard<std::mutex> lk(mu_);
    det_ = r;
  }
  void set_dep(const DepResult& r) {
    std::lock_guard<std::mutex> lk(mu_);
    dep_ = r;
  }
  DetResult det() {
    std::lock_guard<std::mutex> lk(mu_);
    return det_;
  }
  DepResult dep() {
    std::lock_guard<std::mutex> lk(mu_);
    return dep_;
  }

 private:
  std::mutex mu_;
  DetResult det_;
  DepResult dep_;
};

// ---------------------------------------------------------------------------
// Rolling stats
// ---------------------------------------------------------------------------

class Roll {
 public:
  void push(double v) {
    std::lock_guard<std::mutex> lk(mu_);
    win_.push_back(v);
    if (win_.size() > 240) win_.pop_front();
  }
  // avg,min,max,p95
  std::tuple<double, double, double, double> stats() {
    std::lock_guard<std::mutex> lk(mu_);
    if (win_.empty()) return {0, 0, 0, 0};
    std::vector<double> v(win_.begin(), win_.end());
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    return {sum / v.size(), v.front(), v.back(),
            v[(size_t)(0.95 * (v.size() - 1))]};
  }
  std::vector<double> tail(int n) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<double> out;
    int start = (int)win_.size() - n;
    if (start < 0) start = 0;
    for (int i = start; i < (int)win_.size(); ++i) out.push_back(win_[i]);
    return out;
  }

 private:
  std::mutex mu_;
  std::deque<double> win_;
};

// ---------------------------------------------------------------------------
// Compositor + shared JPEG + stats JSON
// ---------------------------------------------------------------------------

struct SharedImage {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<uchar> jpeg;
  uint64_t version = 0;
  void set(const std::vector<uchar>& j) {
    {
      std::lock_guard<std::mutex> lk(mu);
      jpeg = j;
      ++version;
    }
    cv.notify_all();
  }
};

// Composite hand-off between the compositor thread (draw + vconcat) and the
// JPEG encoder thread, so drawing the next frame overlaps encoding the last.
struct CompositeSlot {
  std::mutex mu;
  std::condition_variable cv;
  cv::Mat img;
  uint64_t version = 0;
  void set(const cv::Mat& m) {
    {
      std::lock_guard<std::mutex> lk(mu);
      m.copyTo(img);
      ++version;
    }
    cv.notify_all();
  }
};

// Official Ultralytics Annotator palette (BGR) — first 16 class colors.
static const cv::Scalar kUltra[16] = {
    {56, 56, 255},  {151, 157, 255}, {31, 112, 255},  {29, 178, 255},
    {60, 210, 207}, {10, 249, 72},   {23, 204, 146},  {134, 219, 61},
    {52, 147, 26},  {188, 221, 0},   {255, 194, 0},   {255, 92, 0},
    {255, 0, 43},   {255, 0, 176},   {219, 0, 255},   {128, 0, 255}};

// Ultralytics Annotator-style box: class-colored AA outline + solid filled
// label tag (class color) with auto black/white text; the tag flips below
// the box when it would overflow the top edge — mirrors
// ultralytics.utils.plotting.Annotator.box_label().
static void DrawYoloBox(cv::Mat& im, const Det& d,
                        const std::vector<std::string>& labels) {
  const cv::Scalar& col = kUltra[d.cls % 16];
  const int lw = 3;
  const double sf = 0.80;
  const int tf = std::max(lw, 2);
  const cv::Point p1((int)d.x1, (int)d.y1), p2((int)d.x2, (int)d.y2);
  cv::rectangle(im, p1, p2, col, lw, cv::LINE_AA);

  std::string name = (size_t)d.cls < labels.size() ? labels[d.cls] : "obj";
  char conf[8];
  std::snprintf(conf, sizeof(conf), "%.2f", d.score);
  const std::string label = name + " " + conf;

  int base = 0;
  const cv::Size ts =
      cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, sf, tf, &base);
  // clamp the tag inside the right edge
  int tx = std::max(0, std::min(p1.x, im.cols - ts.width - 2));
  const bool outside = p1.y - ts.height >= 3;  // label fits above the box
  const cv::Point tag2(tx + ts.width,
                       outside ? p1.y - ts.height - 3 : p1.y + ts.height + 3);
  cv::rectangle(im, cv::Point(tx, p1.y), tag2, col, -1, cv::LINE_AA);
  // white text on dark colors, black on bright ones
  const double lum = 0.299 * col[2] + 0.587 * col[1] + 0.114 * col[0];
  const cv::Scalar txtc =
      lum > 150.0 ? cv::Scalar(28, 28, 28) : cv::Scalar(255, 255, 255);
  cv::putText(im, label,
              cv::Point(tx, outside ? p1.y - 2 : p1.y + ts.height + 2),
              cv::FONT_HERSHEY_SIMPLEX, sf, txtc, tf, cv::LINE_AA);
}

// Translucent HUD strip with a colored accent bar.
static void DrawHud(cv::Mat& pane, const std::string& left_txt,
                    const std::string& right_txt, const cv::Scalar& accent) {
  const double sf = 0.80;
  const int tf = 2;
  auto drawOne = [&](const std::string& text, int x, bool right_align) {
    int base = 0;
    cv::Size ts =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, sf, tf, &base);
    if (right_align) x = x - ts.width - 22;
    if (x < 0) x = 0;
    const int y = 10;
    const int w = std::min(ts.width + 26, pane.cols - x);
    const int h = ts.height + 12;
    cv::Rect r(x, y, w, h);
    r &= cv::Rect(0, 0, pane.cols, pane.rows);
    if (r.empty()) return;
    cv::Mat roi = pane(r);
    cv::Mat dark(roi.size(), roi.type(), cv::Scalar(10, 12, 16));
    cv::addWeighted(dark, 0.55, roi, 0.45, 0.0, roi);
    cv::rectangle(pane, cv::Rect(x, y, 3, h), accent, -1);
    cv::putText(pane, text, cv::Point(x + 12, y + ts.height + 4),
                cv::FONT_HERSHEY_SIMPLEX, sf, cv::Scalar(238, 241, 245), tf,
                cv::LINE_AA);
  };
  drawOne(left_txt, 8, false);
  if (!right_txt.empty()) drawOne(right_txt, pane.cols - 8, true);
}

// Vertical Turbo scale bar on the depth pane (near=warm at top, far=cold
// at bottom — matches the colorize math: near -> turbo(255)).
static void DrawDepthScale(cv::Mat& pane) {
  static cv::Mat lut;  // 256x1 turbo column, built once
  if (lut.empty()) {
    cv::Mat ramp(256, 1, CV_8U);
    for (int i = 0; i < 256; ++i) ramp.data[i] = (uchar)(255 - i);
    cv::applyColorMap(ramp, lut, cv::COLORMAP_TURBO);
  }
  const int bar_h = (int)(pane.rows * 0.32);
  const int bar_w = 8;
  const cv::Rect r(pane.cols - bar_w - 12, (pane.rows - bar_h) / 2, bar_w,
                   bar_h);
  cv::Mat bar;
  cv::resize(lut, bar, r.size(), 0, 0, cv::INTER_LINEAR);
  bar.copyTo(pane(r));
  cv::rectangle(pane, r, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
  const double sf = 0.38;
  int base = 0;
  cv::Size ts = cv::getTextSize("NEAR", cv::FONT_HERSHEY_SIMPLEX, sf, 1, &base);
  cv::Point tp(r.x + bar_w / 2 - ts.width / 2, r.y - 6);
  if (tp.y > ts.height) {
    cv::putText(pane, "NEAR", tp, cv::FONT_HERSHEY_SIMPLEX, sf,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(pane, "NEAR", tp, cv::FONT_HERSHEY_SIMPLEX, sf,
                cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
  }
  ts = cv::getTextSize("FAR", cv::FONT_HERSHEY_SIMPLEX, sf, 1, &base);
  cv::Point bp(r.x + bar_w / 2 - ts.width / 2, r.y + r.height + ts.height + 4);
  if (bp.y < pane.rows - 2) {
    cv::putText(pane, "FAR", bp, cv::FONT_HERSHEY_SIMPLEX, sf,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(pane, "FAR", bp, cv::FONT_HERSHEY_SIMPLEX, sf,
                cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
  }
}

// Grid overlay: a cross marker at every grid intersection with the
// distance value of that exact point. rel = 0 (near) .. 1 (far);
// meters_per_unit > 0 switches the display to metric meters.
static void DrawDepthGrid(cv::Mat& pane, const std::vector<float>& grid,
                          int cols, int rows, double meters_per_unit) {
  if (grid.empty() || cols <= 0 || rows <= 0) return;
  const int W = pane.cols, H = pane.rows;
  const int ic = cols + 1, ir = rows + 1;
  if ((int)grid.size() < ic * ir) return;
  // subtle cell borders
  for (int c = 1; c < cols; ++c) {
    const int x = (int)std::lround((double)c * W / cols);
    cv::line(pane, {x, 0}, {x, H}, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
  }
  for (int r = 1; r < rows; ++r) {
    const int y = (int)std::lround((double)r * H / rows);
    cv::line(pane, {0, y}, {W, y}, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
  }
  const double sf = std::max(0.52, std::min(0.70, W / 1100.0));
  const int tf = 1;
  for (int r = 0; r < ir; ++r) {
    for (int c = 0; c < ic; ++c) {
      const float rel = grid[(size_t)r * ic + c];
      const int px = (int)std::lround((double)c * W / cols);
      const int py = (int)std::lround((double)r * H / rows);
      // crosshair marker at the intersection
      const int arm = 4;
      cv::line(pane, {px - arm, py}, {px + arm, py},
               cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
      cv::line(pane, {px, py - arm}, {px, py + arm},
               cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
      // value label offset up-right of the cross, flipped at edges
      const std::string txt =
          ui::FormatFocusDistance(rel, meters_per_unit);
      int base = 0;
      const cv::Size ts =
          cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, sf, tf, &base);
      int tx = px + 9, ty = py - 9;
      if (tx + ts.width > W - 2) tx = px - ts.width - 9;
      if (ty - ts.height < 2) ty = py + ts.height + 9;
      // black outline for readability over the turbo colormap
      cv::putText(pane, txt, {tx, ty}, cv::FONT_HERSHEY_SIMPLEX, sf,
                  cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
      // near = green, mid = amber, far = red-ish
      const cv::Scalar tcol = rel < 0.34f   ? cv::Scalar(70, 225, 90)
                              : rel < 0.67f ? cv::Scalar(240, 215, 100)
                                            : cv::Scalar(90, 120, 255);
      cv::putText(pane, txt, {tx, ty}, cv::FONT_HERSHEY_SIMPLEX, sf, tcol, tf,
                  cv::LINE_AA);
    }
  }
}

// ---------------------------------------------------------------------------
// HDMI native overlay, light theme, resolution adaptive.
//  - LEFT column (under the video): performance gauges (BPU util, latency)
//  - RIGHT column: metric cards + THREE community QR slots
// ---------------------------------------------------------------------------

struct PanelMetrics {
  double display_fps = 0;
  double bpu_util = 0;
  int bpu_freq_mhz = 0;
  int objects = 0;
  double det_avg = 0, det_p95 = 0, det_min = 0, det_max = 0;
  double dep_avg = 0, dep_p95 = 0, dep_min = 0, dep_max = 0;
  double det_fps = 0, dep_fps = 0;
  std::vector<double> det_hist, dep_hist;
};

// light-theme palette (BGR)
static const cv::Scalar kPnlBg(244, 246, 249);      // #F4F6F9 -> #E9EEF4
static const cv::Scalar kPnlCard(255, 255, 255);    // white
static const cv::Scalar kPnlBorder(230, 230, 224);  // soft gray
static const cv::Scalar kPnlText(32, 34, 38);       // near-black
static const cv::Scalar kPnlMuted(148, 148, 138);   // gray
static const cv::Scalar kPnlBlue(255, 146, 64);     // #4092FF
static const cv::Scalar kPnlGreen(126, 199, 42);    // #2AC77E
static const cv::Scalar kPnlAmber(36, 178, 240);    // #F0B224
static const cv::Scalar kPnlRed(64, 84, 255);       // #FF5440

static void PnlRect(cv::Mat& im, const cv::Rect& r, int t = 1) {
  cv::rectangle(im, r, kPnlCard, -1, cv::LINE_AA);
  cv::rectangle(im, r, kPnlBorder, t, cv::LINE_AA);
}

static void PnlText(cv::Mat& im, const std::string& t, const cv::Point& org,
                    double sf, const cv::Scalar& col, int tf = 1) {
  cv::putText(im, t, org, cv::FONT_HERSHEY_SIMPLEX, sf, col, tf, cv::LINE_AA);
}

static std::string PnlFmt(const char* fmt, ...) {
  char b[64];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(b, sizeof(b), fmt, ap);
  va_end(ap);
  return std::string(b);
}

static void PnlSpark(cv::Mat& im, const cv::Rect& r,
                     const std::vector<double>& v, const cv::Scalar& line) {
  if (v.size() < 2) return;
  double mn = v[0], mx = v[0];
  for (double x : v) {
    mn = std::min(mn, x);
    mx = std::max(mx, x);
  }
  const double rng = (mx - mn) > 1e-9 ? (mx - mn) : 1.0;
  const double cw = (double)r.width / (v.size() - 1);
  std::vector<cv::Point> pts;
  for (size_t i = 0; i < v.size(); ++i)
    pts.push_back(cv::Point(r.x + (int)(i * cw),
                            r.y + r.height - 2 -
                                (int)((v[i] - mn) / rng * (r.height - 5))));
  std::vector<cv::Point> poly = pts;
  poly.push_back(cv::Point(r.x + r.width, r.y + r.height - 1));
  poly.push_back(cv::Point(r.x, r.y + r.height - 1));
  cv::fillPoly(im, poly, cv::Scalar(242, 244, 248), cv::LINE_AA);
  cv::line(im, {r.x, r.y + r.height - 1}, {r.x + r.width, r.y + r.height - 1},
           kPnlBorder, 1, cv::LINE_AA);
  cv::polylines(im, pts, false, line, 2, cv::LINE_AA);
}

// Analog gauge: 220° arc from 160° to 380° (CW), colored progress arc,
// value in the center, unit + label under it. drawn into rect r.
static void PnlGauge(cv::Mat& im, const cv::Rect& r, double frac,
                     const std::string& value, const std::string& unit,
                     const std::string& label, const cv::Scalar& accent) {
  const cv::Point c(r.x + r.width / 2, r.y + r.height / 2 + (int)(r.height * 0.12));
  const int rad = (int)std::min(r.width, (int)(r.height * 1.7)) / 2 - 6;
  const double a0 = 200 * CV_PI / 180.0;   // start angle
  const double span = 220 * CV_PI / 180.0; // total sweep
  // background track
  cv::ellipse(im, c, {rad, rad}, 0, 180.0 + 20.0, 180.0 + 20.0 + 220.0,
              kPnlBorder, 6, cv::LINE_AA);
  // value arc (clockwise from the same start)
  const double a1 = a0 + span * std::clamp(frac, 0.0, 1.0);
  const int deg0 = (int)(a0 * 180.0 / CV_PI);
  const int deg1 = (int)(a1 * 180.0 / CV_PI);
  if (deg1 > deg0)
    cv::ellipse(im, c, {rad, rad}, 0, deg0, deg1, accent, 6, cv::LINE_AA);
  // ticks
  for (int t = 0; t <= 4; ++t) {
    const double a = a0 + span * t / 4.0;
    const cv::Point p1(c.x + (int)(cos(a) * (rad - 8)), c.y - (int)(sin(a) * (rad - 8)));
    const cv::Point p2(c.x + (int)(cos(a) * (rad + 2)), c.y - (int)(sin(a) * (rad + 2)));
    cv::line(im, p1, p2, kPnlMuted, 1, cv::LINE_AA);
  }
  // big centered value
  int base = 0;
  cv::Size ts = cv::getTextSize(value, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, &base);
  cv::putText(im, value, {c.x - ts.width / 2, c.y + ts.height / 2 - 2},
              cv::FONT_HERSHEY_SIMPLEX, 0.9, kPnlText, 2, cv::LINE_AA);
  PnlText(im, unit, {c.x - (int)(unit.size() * 8), c.y + (int)(rad * 0.55)},
          0.4, kPnlMuted);
  // label under the gauge
  ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &base);
  PnlText(im, label, {r.x + (r.width - ts.width) / 2, r.y + r.height - 4}, 0.42,
          kPnlText);
}

// RIGHT column: metric cards (bigger fonts). Returns finished panel image.
static cv::Mat DrawSidePanel(const cv::Size& canvas, const PanelMetrics& m,
                             double depth_meters) {
  const int W = canvas.width, H = canvas.height;
  cv::Mat im(canvas, CV_8UC3, kPnlBg);
  const double fs = std::max(1.0, W / 344.0);   // font scale
  const double vs = std::max(1.0, H / 620.0);   // vertical scale
  const int ml = (int)(14 * fs), mr = (int)(14 * fs);
  const int cw = W - ml - mr;

  // header
  PnlText(im, "YOLO26 DETECT + DEPTH", {ml, (int)(26 * vs)}, 0.62 * fs,
          kPnlText, 2);
  PnlText(im, "RDK S100P - hbDNN C++ dual model", {ml, (int)(46 * vs)},
          0.36 * fs, kPnlMuted);
  cv::circle(im, {W - mr - (int)(10 * fs), (int)(22 * vs)},
             (int)(6 * fs) + 1, kPnlGreen, -1, cv::LINE_AA);
  int y = (int)(62 * vs);

  // KPI row
  auto kpi = [&](const cv::Rect& r, const std::string& k, const std::string& v,
                 const cv::Scalar& vc) {
    PnlRect(im, r);
    PnlText(im, k, {r.x + (int)(10 * fs), r.y + (int)(20 * vs)}, 0.36 * fs,
            kPnlMuted);
    int base = 0;
    cv::Size ts =
        cv::getTextSize(v, cv::FONT_HERSHEY_SIMPLEX, 0.95 * fs, 2, &base);
    cv::putText(im, v, {r.x + (r.width - ts.width) / 2,
                        r.y + r.height / 2 + ts.height / 2 + (int)(6 * vs)},
                cv::FONT_HERSHEY_SIMPLEX, 0.95 * fs, vc, 2, cv::LINE_AA);
  };
  {
    const int kpih = (int)(74 * vs);
    const int kpw = (cw - (int)(12 * fs)) / 3;
    kpi({ml, y, kpw, kpih}, "FPS", PnlFmt("%.1f", m.display_fps), kPnlBlue);
    kpi({ml + kpw + (int)(6 * fs), y, kpw, kpih}, "BPU", PnlFmt("%.0f%%", m.bpu_util),
         kPnlAmber);
    kpi({ml + 2 * (kpw + (int)(6 * fs)), y, cw - 2 * (kpw + (int)(6 * fs)),
         kpih},
        "OBJ", PnlFmt("%d", m.objects), kPnlGreen);
    y += kpih + (int)(12 * vs);
  }

  auto kvrow = [&](int x, int yy, const std::string& k, const std::string& v,
                   const cv::Scalar& vc = kPnlText) {
    PnlText(im, k, {x, yy}, 0.36 * fs, kPnlMuted);
    int base = 0;
    cv::Size ts =
        cv::getTextSize(v, cv::FONT_HERSHEY_SIMPLEX, 0.36 * fs, 1, &base);
    PnlText(im, v, {x + (int)(138 * fs) - ts.width, yy}, 0.36 * fs, vc);
  };

  // model cards with sparkline + P95 explained
  auto model_card = [&](const std::string& title, const cv::Scalar& accent,
                        double avg, double p95, double mn, double mx,
                        double fps, const std::vector<double>& hist,
                        const std::string& extra) {
    const int ch = (int)(168 * vs);
    PnlRect(im, {ml, y, cw, ch});
    PnlText(im, title, {ml + (int)(12 * fs), y + (int)(22 * vs)}, 0.4 * fs,
            kPnlMuted);
    PnlText(im, PnlFmt("%.1f fps", fps),
            {ml + cw - (int)(104 * fs), y + (int)(22 * vs)}, 0.4 * fs, accent);
    PnlText(im, PnlFmt("AVG %.2f ms", avg),
            {ml + (int)(12 * fs), y + (int)(54 * vs)}, 0.46 * fs, kPnlText);
    PnlText(im, PnlFmt("P95 %.2f ms", p95),
            {ml + (int)(12 * fs), y + (int)(78 * vs)}, 0.46 * fs, kPnlText);
    PnlText(im, PnlFmt("MIN/MAX %.1f/%.1f", mn, mx),
            {ml + (int)(12 * fs), y + (int)(100 * vs)}, 0.34 * fs, kPnlMuted);
    // P95 note (what it means)
    PnlText(im, "P95: 95% of runs faster than this", {ml + (int)(12 * fs),
            y + (int)(122 * vs)}, 0.26 * fs, kPnlMuted);
    PnlText(im, extra, {ml + (int)(12 * fs), y + (int)(140 * vs)}, 0.3 * fs,
            kPnlMuted);
    PnlSpark(im,
             {ml + cw - (int)(150 * fs), y + (int)(48 * vs),
              (int)(138 * fs), (int)(70 * vs)},
             hist, accent);
    y += ch + (int)(12 * vs);
  };
  model_card("YOLO26x DETECT", kPnlBlue, m.det_avg, m.det_p95, m.det_min,
             m.det_max, m.det_fps, m.det_hist, "640x640 NV12");
  model_card("YOLO26x DEPTH", kPnlGreen, m.dep_avg, m.dep_p95, m.dep_min,
             m.dep_max, m.dep_fps, m.dep_hist,
             depth_meters > 0 ? "768 lite - METERS" : "768 lite - RAW");

  // turbo legend
  {
    const int fh = (int)(40 * vs);
    PnlRect(im, {ml, y, cw, fh});
    cv::Mat ramp(1, cw - (int)(76 * fs), CV_8UC1);
    for (int i = 0; i < ramp.cols; ++i)
      ramp.data[i] = (uchar)(i * 255 / std::max(ramp.cols - 1, 1));
    cv::Mat turbo;
    cv::applyColorMap(ramp, turbo, cv::COLORMAP_TURBO);
    cv::Mat roi = im(cv::Rect(ml + (int)(38 * fs), y + fh / 2 - (int)(6 * vs),
                              ramp.cols, (int)(12 * vs)));
    cv::resize(turbo, roi, roi.size(), 0, 0, cv::INTER_LINEAR);
    PnlText(im, "NEAR", {ml + (int)(2 * fs), y + fh / 2 + (int)(5 * vs)},
            0.3 * fs, kPnlMuted);
    PnlText(im, "FAR", {ml + cw - (int)(34 * fs), y + fh / 2 + (int)(5 * vs)},
            0.3 * fs, kPnlMuted);
  }
  return im;
}

// LEFT column (below the video): two big gauges — BPU load and total latency.
static cv::Mat DrawGaugeColumn(const cv::Size& canvas, const PanelMetrics& m) {
  cv::Mat im(canvas, CV_8UC3, kPnlBg);
  const int W = canvas.width, H = canvas.height;
  const double fs = std::max(1.0, W / 300.0);
  const int gw = W - 16;
  const int gh = (int)(H * 0.46) - 10;
  const cv::Scalar lc = m.bpu_util < 60   ? kPnlGreen
                        : m.bpu_util < 85 ? kPnlAmber
                                          : kPnlRed;
  // card 1: BPU gauge
  PnlRect(im, {4, 4, gw + 8, gh}, 2);
  PnlText(im, "BPU LOAD", {12, (int)(26 * std::max(1.0, H / 300.0))},
          0.5 * fs, kPnlMuted);
  PnlGauge(im, {4, 4, gw + 8, gh}, m.bpu_util / 100.0,
           PnlFmt("%.1f", m.bpu_util), "%", "BPU UTIL", lc);
  // card 2: latency gauge (0..30ms range)
  const int y2 = gh + 14;
  const int gh2 = H - y2 - 4;
  PnlRect(im, {4, y2, gw + 8, gh2}, 2);
  PnlText(im, "INFER LATENCY / PAIR", {12, y2 + (int)(22 * std::max(1.0, H / 300.0))},
          0.5 * fs, kPnlMuted);
  const double pair = m.det_avg + m.dep_avg;
  PnlGauge(im, {4, y2, gw + 8, gh2},
           std::clamp(pair / 30.0, 0.0, 1.0), PnlFmt("%.1f", pair), "ms",
           "DET+DEP", pair < 25.0 ? kPnlGreen : kPnlAmber);
  return im;
}

// THREE community QR slots (stacked) for the right column bottom.
static void DrawQrColumn(cv::Mat& canvas, const cv::Point& org, int total_w,
                         int total_h) {
  // three white cards stacked vertically, each ~46% height
  const int n = 3;
  const int gap = (int)(total_h * 0.03);
  const int card_h = (total_h - gap * (n - 1)) / n;
  static cv::Mat qr[3];
  const char* paths[3] = {
      "/userdata/yolo26_dual_demo/assets/qr1.png",
      "/userdata/yolo26_dual_demo/assets/qr2.png",
      "/userdata/yolo26_dual_demo/assets/qr3.png"};
  static bool loaded = false;
  if (!loaded) {
    for (int i = 0; i < n; ++i)
      qr[i] = cv::imread(paths[i], cv::IMREAD_COLOR);
    loaded = true;
  }
  for (int i = 0; i < n; ++i) {
    const cv::Rect r(org.x, org.y + i * (card_h + gap), total_w, card_h);
    PnlRect(canvas, r, 2);
    const int pad = (int)(r.width * 0.08);
    const int side = std::min(r.width - 2 * pad, r.height - (int)(r.height * 0.22) - 2 * pad);
    const cv::Rect inner(r.x + (r.width - side) / 2, r.y + pad, side, side);
    if (!qr[i].empty()) {
      cv::Mat dst = canvas(inner);
      cv::Mat rs;
      cv::resize(qr[i], rs, inner.size(), 0, 0,
                 qr[i].cols > inner.width ? cv::INTER_AREA : cv::INTER_LINEAR);
      rs.copyTo(dst);
    } else {
      cv::rectangle(canvas, inner, cv::Scalar(228, 232, 238), -1, cv::LINE_AA);
      for (int gy = 0; gy < 3; ++gy)
        for (int gx = 0; gx < 3; ++gx)
          if ((gx + gy) % 2 == 0)
            cv::rectangle(canvas,
                          cv::Rect(inner.x + gx * side / 3 + 2,
                                   inner.y + gy * side / 3 + 2,
                                   side / 3 - 4, side / 3 - 4),
                          kPnlMuted, -1, cv::LINE_AA);
      // caption
      const char* names[3] = {"COMMUNITY", "DOCS", "GITHUB"};
      int base = 0;
      cv::Size ts = cv::getTextSize(names[i], cv::FONT_HERSHEY_SIMPLEX, 0.34, 1, &base);
      PnlText(canvas, names[i],
              {r.x + (r.width - ts.width) / 2, r.y + r.height - (int)(r.height * 0.08)},
              0.34, kPnlText);
    }
  }
}

// Serve the dashboard from an external file so the UI can be edited
// without recompiling the C++ binary. Fallback: a minimal inline page.
static std::string LoadIndexHtml() {
  const char* paths[] = {"web/index.html",
                         "/userdata/yolo26_dual_demo/web/index.html"};
  for (const char* path : paths) {
    FILE* f = std::fopen(path, "rb");
    if (f) {
      std::fseek(f, 0, SEEK_END);
      long sz = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      std::string s((size_t)sz, '\0');
      size_t rd = std::fread(&s[0], 1, (size_t)sz, f);
      std::fclose(f);
      if (rd == (size_t)sz) return s;
    }
  }
  return
      "<!doctype html><html><head><meta charset=utf-8>"
      "<title>YOLO26 Dual</title><style>body{background:#111;color:#eee;"
      "font-family:sans-serif;text-align:center}img{max-width:100%;max-height:96vh}"
      "</style></head><body><h3>RDK S100P - YOLO26 Detect + Depth</h3>"
      "<img src='/stream'><p><a href='/stats.json'>stats</a></p></body></html>";
}

// ---------------------------------------------------------------------------
// HTTP server (MJPEG + snapshot + stats + index)
// ---------------------------------------------------------------------------

class HttpServer {
 public:
  HttpServer(int port, SharedImage* img, std::function<std::string()> stats)
      : port_(port), img_(img), stats_fn_(std::move(stats)) {}

  bool start() {
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) return false;
    int one = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port_);
    if (bind(fd_, (sockaddr*)&addr, sizeof(addr)) < 0) return false;
    if (listen(fd_, 16) < 0) return false;
    accept_thread_ = std::thread([this] { acceptLoop(); });
    return true;
  }
  void stop() {
    g_stop = true;
    if (fd_ >= 0) {
      shutdown(fd_, SHUT_RDWR);
      close(fd_);
    }
    if (accept_thread_.joinable()) accept_thread_.join();
  }

 private:
  void acceptLoop() {
    while (!g_stop.load()) {
      int c = accept(fd_, nullptr, nullptr);
      if (c < 0) break;
      std::thread([this, c] { serve(c); }).detach();
    }
  }
  static bool sendAll(int c, const void* data, size_t len) {
    const char* p = (const char*)data;
    size_t off = 0;
    while (off < len) {
      ssize_t n = ::send(c, p + off, len - off, MSG_NOSIGNAL);
      if (n <= 0) return false;
      off += (size_t)n;
    }
    return true;
  }
  void respond(int c, const std::string& ctype, const std::string& body) {
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: " + ctype +
                       "\r\nContent-Length: " + std::to_string(body.size()) +
                       "\r\nConnection: close\r\n\r\n";
    sendAll(c, head.data(), head.size());
    sendAll(c, body.data(), body.size());
  }
  void serve(int c) {
    // read request head
    std::string req;
    char buf[2048];
    while (req.find("\r\n\r\n") == std::string::npos) {
      ssize_t n = recv(c, buf, sizeof(buf), 0);
      if (n <= 0) {
        close(c);
        return;
      }
      req.append(buf, (size_t)n);
    }
    // first line: METHOD PATH
    std::string path = req.substr(4, req.find(' ', 4) - 4);
    if (path.rfind('/') == 0 && path.size() > 1) path = path.substr(0, path.find('?'));
    if (path == "/stream") {
      std::string head =
          "HTTP/1.1 200 OK\r\nContent-Type: "
          "multipart/x-mixed-replace; boundary=frame\r\n\r\n";
      if (!sendAll(c, head.data(), head.size())) {
        close(c);
        return;
      }
      uint64_t last_ver = 0;
      while (!g_stop.load()) {
        std::vector<uchar> jpeg;
        {
          std::unique_lock<std::mutex> lk(img_->mu);
          img_->cv.wait_for(lk, std::chrono::milliseconds(100), [&] {
            return img_->version > last_ver || g_stop.load();
          });
          if (img_->version <= last_ver) continue;
          last_ver = img_->version;
          jpeg = img_->jpeg;
        }
        std::string part = std::string("--frame\r\nContent-Type: image/jpeg\r\n") +
                           "Content-Length: " + std::to_string(jpeg.size()) +
                           "\r\n\r\n";
        if (!sendAll(c, part.data(), part.size()) ||
            !sendAll(c, jpeg.data(), jpeg.size()) ||
            !sendAll(c, "\r\n", 2)) {
          break;
        }
      }
      close(c);
    } else if (path == "/snapshot.jpg") {
      std::vector<uchar> jpeg;
      {
        std::lock_guard<std::mutex> lk(img_->mu);
        jpeg = img_->jpeg;
      }
      if (jpeg.empty()) {
        std::string e = "HTTP/1.1 503\r\nConnection: close\r\n\r\n";
        sendAll(c, e.data(), e.size());
      } else {
        respond(c, "image/jpeg", std::string((char*)jpeg.data(), jpeg.size()));
      }
      close(c);
    } else if (path == "/stats.json") {
      respond(c, "application/json", stats_fn_());
      close(c);
    } else if (path == "/stop") {
      respond(c, "text/plain", "stopping\n");
      g_stop = true;
      close(c);
    } else if (path == "/" || path.empty()) {
      respond(c, "text/html; charset=utf-8", LoadIndexHtml());  // hot reload
      close(c);
    } else if (path.rfind("/qr/", 0) == 0) {
      static const std::string qr_dir = "/userdata/yolo26_dual_demo/web/";
      std::string name = path.substr(4);
      // reject path traversal and keep only basenames
      const bool ok_name =
          name.find('/') == std::string::npos &&
          name.find("..") == std::string::npos && name.size() > 4;
      const bool png = ok_name && name.substr(name.size() - 4) == ".png";
      const bool jpg = ok_name && name.substr(name.size() - 4) == ".jpg";
      if (png || jpg) {
      std::string file = qr_dir + name;
      std::fprintf(stderr, "[qr ] path='%s' -> file='%s'\n", path.c_str(),
                   file.c_str());
        FILE* f = std::fopen(file.c_str(), "rb");
        if (f) {
          std::fseek(f, 0, SEEK_END);
          long sz = std::ftell(f);
          std::fseek(f, 0, SEEK_SET);
          std::string body((size_t)sz, '\0');
          size_t rd = std::fread(&body[0], 1, (size_t)sz, f);
          std::fclose(f);
          if (rd == (size_t)sz) {
            const std::string ct = (file.substr(file.size() - 4) == ".png")
                                       ? "image/png"
                                       : "image/jpeg";
            respond(c, ct, body);
            close(c);
            return;
          }
        }
      }
      std::string e = "HTTP/1.1 404\r\nConnection: close\r\n\r\n";
      sendAll(c, e.data(), e.size());
      close(c);
    } else {
      std::string e = "HTTP/1.1 404\r\nConnection: close\r\n\r\n";
      sendAll(c, e.data(), e.size());
      close(c);
    }
  }

  int port_;
  int fd_ = -1;
  SharedImage* img_;
  std::function<std::string()> stats_fn_;
  std::thread accept_thread_;

};

// ---------------------------------------------------------------------------
// Index page: adaptive dual-pane video + live parameter side panel
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

struct Args {
  std::string det_model, dep_model, dep_variant = "x";
  int grid_cols = 8, grid_rows = 6;
  bool no_grid = false;
  double depth_meters = 0;  // 0 = raw normalized; >0 = meters calibration
  std::string source = "0";
  int cam_w = 1280, cam_h = 720, cam_fps = 30;
  int port = 8080;
  float score = 0.25f, nms = 0.45f;
  float track_thresh = 0.30f;
  float track_low_thresh = 0.10f;
  float track_match_thresh = 0.80f;
  int track_buffer = 60;
  int prio_det = 8, prio_dep = 4;
  int bpu_core = 0;
  int warmup = 3;
  double max_seconds = 0;
  std::string save;
  std::string labels = "assets/coco_classes.names";
  std::string image_mode;
  bool hdmi = false;
};

static Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      return (i + 1 < argc) ? argv[++i] : "";
    };
    if (k == "--det-model") a.det_model = next();
    else if (k == "--dep-model") a.dep_model = next();
    else if (k == "--dep-variant") a.dep_variant = next();
    else if (k == "--grid-cols") a.grid_cols = std::stoi(next());
    else if (k == "--grid-rows") a.grid_rows = std::stoi(next());
    else if (k == "--no-grid") a.no_grid = true;
    else if (k == "--depth-meters") a.depth_meters = std::stod(next());
    else if (k == "--source") a.source = next();
    else if (k == "--cam-w") a.cam_w = std::stoi(next());
    else if (k == "--cam-h") a.cam_h = std::stoi(next());
    else if (k == "--cam-fps") a.cam_fps = std::stoi(next());
    else if (k == "--port") a.port = std::stoi(next());
    else if (k == "--score") a.score = std::stof(next());
    else if (k == "--nms") a.nms = std::stof(next());
    else if (k == "--track-thresh") a.track_thresh = std::stof(next());
    else if (k == "--track-low-thresh")
      a.track_low_thresh = std::stof(next());
    else if (k == "--track-match-thresh")
      a.track_match_thresh = std::stof(next());
    else if (k == "--track-buffer") a.track_buffer = std::stoi(next());
    else if (k == "--priority-det") a.prio_det = std::stoi(next());
    else if (k == "--priority-dep") a.prio_dep = std::stoi(next());
    else if (k == "--bpu-core") a.bpu_core = std::stoi(next());
    else if (k == "--warmup") a.warmup = std::stoi(next());
    else if (k == "--max-seconds") a.max_seconds = std::stod(next());
    else if (k == "--save") a.save = next();
    else if (k == "--labels") a.labels = next();
    else if (k == "--image") a.image_mode = next();
    else if (k == "--hdmi") a.hdmi = true;
    else {
      std::fprintf(stderr, "unknown arg: %s\n", k.c_str());
      std::exit(2);
    }
  }
  return a;
}

static std::vector<std::string> LoadLabels(const std::string& path) {
  std::vector<std::string> labels;
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return labels;
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (!s.empty()) labels.push_back(s);
  }
  std::fclose(f);
  return labels;
}

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);
  tracking::Config tracking_config;
  tracking_config.high_thresh = args.track_thresh;
  tracking_config.low_thresh = args.track_low_thresh;
  tracking_config.match_thresh = args.track_match_thresh;
  tracking_config.track_buffer = args.track_buffer;
  const std::string tracking_error = tracking::ValidateConfig(tracking_config);
  if (!tracking_error.empty()) {
    std::fprintf(stderr, "[FATAL] %s\n", tracking_error.c_str());
    return 2;
  }
  tracking::ByteTracker tracker(tracking_config);
  std::signal(SIGINT, [](int) { g_stop = true; });
  std::signal(SIGTERM, [](int) { g_stop = true; });

  // default models (largest variants that saturate the BPU while keeping
  // 30 fps end-to-end on S100P: detect x 9.6 ms + depth x lite 13.6 ms)
  if (args.det_model.empty())
    args.det_model = "models/yolo26x_detect_nashm_640x640_nv12.hbm";
  if (args.dep_model.empty()) {
    if (args.dep_variant == "l" || args.dep_variant == "x")
      args.dep_model = "models/yolo26" + args.dep_variant +
                       "_depth_lite_nashm_768x768.hbm";
    else
      args.dep_model = "models/yolo26" + args.dep_variant +
                       "_depth_nashm_768x768_nv12.hbm";
  }

  // ---- load both models into one packed DNN handle -----------------------
  const char* files[2] = {args.det_model.c_str(), args.dep_model.c_str()};
  hbDNNPackedHandle_t packed = nullptr;
  HB_CHECK(hbDNNInitializeFromFiles(&packed, files, 2),
           "hbDNNInitializeFromFiles");
  char const** names = nullptr;
  int32_t n_names = 0;
  HB_CHECK(hbDNNGetModelNameList(&names, &n_names, packed),
           "hbDNNGetModelNameList");
  if (n_names != 2) {
    std::fprintf(stderr, "[FATAL] expected 2 models, got %d\n", n_names);
    return 1;
  }
  std::printf("[dnn  ] loaded: %s + %s\n", names[0], names[1]);
  std::printf("[dnn  ] version=%s ucp=%s soc=%s\n", hbDNNGetVersion(),
              hbUCPGetVersion(), hbUCPGetSocName());

  // Identify models by IO signature: the YOLO26 detect graph has 6 outputs
  // (3 scales x cls/box), the depth graph has 1 output. The name-list order
  // does NOT follow the file order, so match by signature.
  std::string det_name, dep_name;
  for (int32_t i = 0; i < n_names; ++i) {
    hbDNNHandle_t h = nullptr;
    hbDNNGetModelHandle(&h, packed, names[i]);
    int32_t oc = 0;
    hbDNNGetOutputCount(&oc, h);
    if (oc >= 4)
      det_name = names[i];
    else
      dep_name = names[i];
  }
  if (det_name.empty() || dep_name.empty()) {
    std::fprintf(stderr, "[FATAL] could not identify detect/depth models\n");
    return 1;
  }

  std::vector<std::string> labels = LoadLabels(args.labels);

  DetectModel det(packed, det_name, labels);
  DepthModel dep(packed, dep_name, args.dep_variant);
  if (!args.no_grid && args.grid_cols > 0 && args.grid_rows > 0)
    dep.set_grid(args.grid_cols, args.grid_rows);
  std::printf("[model] detect %s input=%dx%d nv12\n", det.model_name().c_str(),
              det.input_size(), det.input_size());
  std::printf("[model] depth  %s input=%dx%d profile=%s\n",
              dep.model_name().c_str(), dep.input_size(), dep.input_size(),
              dep.profile().c_str());

  const uint64_t backend = 1ULL << args.bpu_core;

  // ---- image mode ---------------------------------------------------------
  if (!args.image_mode.empty()) {
    cv::Mat img = cv::imread(args.image_mode, cv::IMREAD_COLOR);
    if (img.empty()) {
      std::fprintf(stderr, "[FATAL] cannot read %s\n", args.image_mode.c_str());
      return 1;
    }
    for (int i = 0; i < args.warmup; ++i) {
      det.Preprocess(img);
      det.Infer(args.prio_det, backend);
      dep.Preprocess(img);
      dep.Infer(args.prio_dep, backend);
    }
    double pre1 = det.Preprocess(img);
    double bpu1 = det.Infer(args.prio_det, backend);
    DetResult dr;
    dr.src_w = img.cols;
    dr.src_h = img.rows;
    std::vector<tracking::SourceDetection> source_dets;
    double post1 = det.Postprocess(
        img.cols, img.rows, std::min(args.score, args.track_low_thresh),
        args.score, args.nms, dr.dets, source_dets, dr.cls_hist);
    dr.tracks = tracker.Update(tracking::SelectPersonDetections(
        source_dets, args.track_low_thresh));
    dr.object_count = tracking::CountDisplayedObjects(
        static_cast<int>(dr.tracks.size()), source_dets, args.score);
    double pre2 = dep.Preprocess(img);
    double bpu2 = dep.Infer(args.prio_dep, backend);
    cv::Mat depth_color;
    double post2 = dep.Postprocess(img, depth_color);

    // composite
    cv::Mat top = img.clone();
    for (const auto& d : dr.dets) {
      if (tracking::ShouldRenderRawDetection(d.cls)) {
        DrawYoloBox(top, d, labels);
      }
    }
    for (const auto& track : dr.tracks) ui::DrawTrackBox(top, track);
    DrawHud(top, "YOLO26x DETECT + PERSON TRACKING 640x640 NV12 | BPU " +
                     std::to_string(bpu1).substr(0, 5) + " ms | " +
                     std::to_string(dr.object_count) + " objs",
            "input " + std::to_string(img.cols) + "x" +
                std::to_string(img.rows),
            kUltra[2]);
    DrawDepthScale(depth_color);
    DrawDepthGrid(depth_color, dep.grid_vals(), dep.grid_cols(),
                  dep.grid_rows(), args.depth_meters);
    DrawHud(depth_color, "YOLO26x DEPTH DISTANCE GRID 768x768 lite | BPU " +
                             std::to_string(bpu2).substr(0, 5) +
                             " ms | Turbo",
            "near warm / far cold", kUltra[6]);
    cv::Mat divider(4, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat composite;
    cv::vconcat(top, divider, composite);
    cv::vconcat(composite, depth_color, composite);
    std::string out = args.save.empty() ? "output/composite_cpp.jpg" : args.save;
    cv::imwrite(out, composite);
    std::printf("{\n  \"output\": \"%s\",\n  \"objects\": %d,\n"
                "  \"det\": {\"pre_ms\":%.2f, \"bpu_ms\":%.2f, \"post_ms\":%.2f},\n"
                "  \"depth\": {\"pre_ms\":%.2f, \"bpu_ms\":%.2f, \"post_ms\":%.2f}\n}\n",
                out.c_str(), dr.object_count, pre1, bpu1, post1, pre2, bpu2,
                post2);
    hbDNNRelease(packed);
    return 0;
  }

  // ---- camera / video mode ------------------------------------------------
  cv::VideoCapture cap;
  // Any pure-numeric source is a V4L2 camera index (0,1,2,3,...).
  const bool is_cam_idx = !args.source.empty() &&
      std::all_of(args.source.begin(), args.source.end(),
                  [](unsigned char c) { return std::isdigit(c); });
  if (is_cam_idx) {
    // boot-race safety: the USB camera may enumerate seconds after the
    // service starts — retry instead of failing
    for (int attempt = 1; attempt <= 10; ++attempt) {
      cap.open(std::stoi(args.source), cv::CAP_V4L2);
      if (cap.isOpened()) break;
      std::printf("[camera] open attempt %d/10 failed, retrying...\n",
                  attempt);
      for (int w = 0; w < 30 && !g_stop.load(); ++w)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (g_stop.load()) break;
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, args.cam_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, args.cam_h);
    cap.set(cv::CAP_PROP_FPS, args.cam_fps);
  } else {
    cap.open(args.source);
  }
  if (!cap.isOpened()) {
    std::fprintf(stderr, "[FATAL] cannot open source %s\n",
                 args.source.c_str());
    return 1;
  }
  std::printf("[camera] %dx%d @ %.1f fps\n", (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
              (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT),
              cap.get(cv::CAP_PROP_FPS));

  FrameHub hub;
  ResultStore store;
  SharedImage shared;
  Roll det_roll, dep_roll, track_roll;
  std::atomic<uint64_t> det_count{0}, dep_count{0}, cam_count{0};
  std::atomic<double> det_pre_ms{0}, det_post_ms{0};
  std::atomic<double> dep_pre_ms{0}, dep_post_ms{0};
  std::mutex bpu_mu;
  std::deque<std::pair<double, double>> det_events, dep_events;  // (t, bpu_ms)
  std::atomic<double> display_fps{0};
  fclock::time_point t_start = NOW();

  // rolling BPU load over last 5s
  auto bpu_load = [&]() -> double {
    // Read the real BPU utilization from the sysfs ratio device (0-100 %).
    // This reflects actual hardware occupancy instead of the previous
    // estimate from event latencies (which over-reports).
    FILE* f = std::fopen("/sys/devices/system/bpu/ratio", "r");
    if (f) {
      char buf[64] = {0};
      if (std::fgets(buf, sizeof(buf), f)) {
        std::fclose(f);
        double v = std::atof(buf);
        if (v >= 0.0 && v <= 100.0) return v;
      } else {
        std::fclose(f);
      }
    }
    // fallback: previous event-latency estimate
    double now = ms_since(t_start) / 1000.0;
    auto prune = [&](std::deque<std::pair<double, double>>& q) {
      while (!q.empty() && now - q.front().first > 5.0) q.pop_front();
    };
    double s = 0;
    {
      std::lock_guard<std::mutex> lk(bpu_mu);
      prune(det_events);
      prune(dep_events);
      for (auto& e : det_events) s += e.second;
      for (auto& e : dep_events) s += e.second;
    }
    return s / 50.0;
  };

  // ---- workers -------------------------------------------------------------
  std::thread det_worker([&] {
    cv::Mat frame;
    uint64_t seq = 0;
    // warmup
    for (int i = 0; i < args.warmup; ++i) {
      det.Preprocess(cv::Mat(args.cam_h / 2, args.cam_w / 2, CV_8UC3,
                             cv::Scalar(0, 0, 0)));
      det.Infer(args.prio_det, backend);
    }
    std::printf("[detect] ready (%s)\n", det.model_name().c_str());
    while (!g_stop.load()) {
      if (!hub.latest(seq, frame, 1.0)) continue;
      DetResult r;
      r.src_w = frame.cols;
      r.src_h = frame.rows;
      r.pre_ms = det.Preprocess(frame);
      r.bpu_ms = det.Infer(args.prio_det, backend);
      std::vector<tracking::SourceDetection> source_dets;
      r.post_ms = det.Postprocess(
          frame.cols, frame.rows,
          std::min(args.score, args.track_low_thresh), args.score, args.nms,
          r.dets, source_dets, r.cls_hist);
      auto person_inputs = tracking::SelectPersonDetections(
          source_dets, args.track_low_thresh);
      auto track_t0 = NOW();
      try {
        r.tracks = tracker.Update(person_inputs);
      } catch (const std::exception& error) {
        std::fprintf(stderr, "[track] update failed: %s\n", error.what());
        r.tracks.clear();
      } catch (...) {
        std::fprintf(stderr, "[track] update failed: unknown exception\n");
        r.tracks.clear();
      }
      r.tracking_ms = ms_since(track_t0);
      try {
        r.tracker_stats = tracker.Snapshot();
      } catch (const std::exception& error) {
        std::fprintf(stderr, "[track] snapshot failed: %s\n", error.what());
        r.tracker_stats = {};
      } catch (...) {
        std::fprintf(stderr, "[track] snapshot failed: unknown exception\n");
        r.tracker_stats = {};
      }
      r.tracker_stats.active = static_cast<int>(r.tracks.size());
      r.cls_hist[0] = r.tracker_stats.active;
      r.object_count = tracking::CountDisplayedObjects(
          r.tracker_stats.active, source_dets, args.score);
      r.stamp = ms_since(t_start) / 1000.0;
      det_pre_ms.store(r.pre_ms);
      det_post_ms.store(r.post_ms);
      det_count.fetch_add(1);
      det_roll.push(r.bpu_ms);
      track_roll.push(r.tracking_ms);
      {
        std::lock_guard<std::mutex> lk(bpu_mu);
        det_events.push_back({ms_since(t_start) / 1000.0, r.bpu_ms});
      }
      store.set_det(r);
    }
  });

  std::thread dep_worker([&] {
    cv::Mat frame;
    uint64_t seq = 0;
    for (int i = 0; i < args.warmup; ++i) {
      dep.Preprocess(cv::Mat(args.cam_h / 2, args.cam_w / 2, CV_8UC3,
                             cv::Scalar(0, 0, 0)));
      dep.Infer(args.prio_dep, backend);
    }
    std::printf("[depth ] ready (%s)\n", dep.model_name().c_str());
    while (!g_stop.load()) {
      if (!hub.latest(seq, frame, 1.0)) continue;
      DepResult r;
      r.pre_ms = dep.Preprocess(frame);
      r.bpu_ms = dep.Infer(args.prio_dep, backend);
      r.post_ms = dep.Postprocess(frame, r.color);
      r.grid = dep.grid_vals();
      r.grid_cols = dep.grid_cols();
      r.grid_rows = dep.grid_rows();
      r.stamp = ms_since(t_start) / 1000.0;
      dep_pre_ms.store(r.pre_ms);
      dep_post_ms.store(r.post_ms);
      dep_count.fetch_add(1);
      dep_roll.push(r.bpu_ms);
      {
        std::lock_guard<std::mutex> lk(bpu_mu);
        dep_events.push_back({ms_since(t_start) / 1000.0, r.bpu_ms});
      }
      store.set_dep(r);
    }
  });

  // ---- compositor + encoder pipeline ---------------------------------------
  cv::VideoWriter writer;
  CompositeSlot comp_slot;
  std::thread encoder([&] {
    uint64_t last_ver = 0;
    while (!g_stop.load()) {
      cv::Mat composite;
      {
        std::unique_lock<std::mutex> lk(comp_slot.mu);
        comp_slot.cv.wait_for(lk, std::chrono::milliseconds(100), [&] {
          return comp_slot.version > last_ver || g_stop.load();
        });
        if (comp_slot.version <= last_ver) continue;
        last_ver = comp_slot.version;
        composite = comp_slot.img;
      }
      if (composite.empty()) continue;
      std::vector<uchar> jpeg;
      cv::imencode(".jpg", composite, jpeg, {cv::IMWRITE_JPEG_QUALITY, 85});
      shared.set(jpeg);
    }
  });
  std::thread composer([&] {
    auto next_tick = NOW();
    uint64_t last_frame_seq = 0;
    double last_det_stamp = -1, last_dep_stamp = -1;
    int fps_counter = 0;
    auto fps_t0 = NOW();
    cv::Mat frame;
    while (!g_stop.load()) {
      // 20 ms cadence with new-content gating so display can track the full
      // 30 fps camera rate (idle ticks cost nothing)
      next_tick += std::chrono::milliseconds(20);
      std::this_thread::sleep_until(next_tick);
      uint64_t fseq = hub.seq();
      DetResult dr = store.det();
      DepResult dpr = store.dep();
      bool new_content = (fseq != last_frame_seq) ||
                         (dr.stamp != last_det_stamp) ||
                         (dpr.stamp != last_dep_stamp);
      if (!new_content) continue;
      last_frame_seq = fseq;
      last_det_stamp = dr.stamp;
      last_dep_stamp = dpr.stamp;
      frame = hub.get();
      if (frame.empty()) continue;
      cv::Mat top = frame.clone();
      for (const auto& d : dr.dets) {
        if (tracking::ShouldRenderRawDetection(d.cls)) {
          DrawYoloBox(top, d, labels);
        }
      }
      for (const auto& track : dr.tracks) ui::DrawTrackBox(top, track);
      cv::Mat bottom = (dpr.color.empty())
                           ? cv::Mat(frame.rows, frame.cols, CV_8UC3,
                                     cv::Scalar(0, 0, 0))
                           : dpr.color.clone();
      auto fmt = [](double v) {
        char b[16];
        std::snprintf(b, sizeof(b), "%5.1f", v);
        return std::string(b);
      };
      DrawHud(top,
              "YOLO26x DETECT + PERSON TRACKING 640 NV12 | BPU " +
                  fmt(dr.bpu_ms) + " ms | " +
                  std::to_string(dr.object_count) + " objs",
              "cam " + std::to_string(frame.cols) + "x" +
                  std::to_string(frame.rows) + " | " +
                  std::to_string(display_fps.load()).substr(0, 4) + " FPS",
              kUltra[2]);
      DrawDepthScale(bottom);
      DrawDepthGrid(bottom, dpr.grid, dpr.grid_cols, dpr.grid_rows,
                    args.depth_meters);
      DrawHud(bottom, "YOLO26x DEPTH DISTANCE GRID 768 lite | BPU " +
                          fmt(dpr.bpu_ms) + " ms | Turbo",
              "near warm / far cold", kUltra[6]);
      // Stack vertically (top: detect, bottom: depth), compress each pane to
      // a landscape-ish aspect ratio, letterboxing to the canvas width so the
      // image is NOT stretched (keeps the source aspect ratio).
      const int pane_h = 420;                // compressed pane height
      const int canvas_w = frame.cols;       // 1280
      cv::Mat divider(4, canvas_w, CV_8UC3, cv::Scalar(80, 80, 80));
      auto letterbox = [&](const cv::Mat& src, cv::Mat& out) {
        if (src.empty()) { out = cv::Mat(pane_h, canvas_w, CV_8UC3, cv::Scalar(0,0,0)); return; }
        double scale = std::min((double)canvas_w/src.cols, (double)pane_h/src.rows);
        int w = std::max(1, (int)(src.cols*scale)), h = std::max(1, (int)(src.rows*scale));
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(w,h));
        out = cv::Mat(pane_h, canvas_w, CV_8UC3, cv::Scalar(16,18,22));
        int x = (canvas_w-w)/2, y = (pane_h-h)/2;
        resized.copyTo(out(cv::Rect(x, y, w, h)));
      };
      cv::Mat left, right;
      cv::Mat composite;
      letterbox(top, left);
      letterbox(bottom, right);
      cv::vconcat(left, divider, composite);
      cv::vconcat(composite, right, composite);
      // hand off to the encoder thread (JPEG encode ~20ms overlaps the
      // next frame's drawing)
      comp_slot.set(composite);
      if (!args.save.empty()) {
        if (!writer.isOpened()) {
          writer.open(args.save, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                      30.0, cv::Size(composite.cols, composite.rows));
          std::printf("[save ] writing %s\n", args.save.c_str());
        }
        writer.write(composite);
      }
      ++fps_counter;
      double dt = ms_since(fps_t0) / 1000.0;
      if (dt >= 1.0) {
        display_fps.store(fps_counter / dt);
        fps_counter = 0;
        fps_t0 = NOW();
      }
    }
    if (writer.isOpened()) writer.release();
  });

  // ---- HDMI KMS output -------------------------------------------------------
  KmsDisplay kms;
  std::thread hdmi_thread;
  std::atomic<uint64_t> hdmi_flips{0};
  std::atomic<int> hdmi_ready{0};
  std::mutex panel_mu;
  std::vector<double> panel_det_hist, panel_dep_hist;
  if (args.hdmi) {
    hdmi_thread = std::thread([&] {
      uint64_t last_ver = 0;
      cv::Mat panel_cache, gauge_cache;
      cv::Size panel_size(0, 0), gauge_size(0, 0);
      auto build_caches = [&]() {
        PanelMetrics pm;
        {
          auto [da, dmn, dmx, dp] = det_roll.stats();
          auto [pa, pmn, pmx, pp] = dep_roll.stats();
          pm.det_avg = da; pm.det_min = dmn; pm.det_max = dmx; pm.det_p95 = dp;
          pm.dep_avg = pa; pm.dep_min = pmn; pm.dep_max = pmx; pm.dep_p95 = pp;
        }
        pm.display_fps = display_fps.load();
        pm.bpu_util = bpu_load();
        pm.bpu_freq_mhz = 1500;
        pm.objects = store.det().object_count;
        pm.det_fps = det_count.load() / std::max(ms_since(t_start) / 1000.0 - 2.0, 1.0);
        pm.dep_fps = dep_count.load() / std::max(ms_since(t_start) / 1000.0 - 2.0, 1.0);
        {
          std::lock_guard<std::mutex> lk(panel_mu);
          pm.det_hist = panel_det_hist;
          pm.dep_hist = panel_dep_hist;
        }
        if (kms.Ready()) {
          // right panel ~1/6 width (320..560); left gauge column ~300px
          const int pw = std::max(320, std::min(560, kms.Width() / 6));
          const int gw = std::min(340, (int)(kms.Width() * 0.16));
          panel_size = cv::Size(pw, kms.Height());
          gauge_size = cv::Size(gw, (int)(kms.Height() * 0.42));
          panel_cache = DrawSidePanel(panel_size, pm, args.depth_meters);
          gauge_cache = DrawGaugeColumn(gauge_size, pm);
        }
      };
      int panel_tick = 0;
      while (!g_stop.load()) {
        if (!kms.Ready()) {
          if (!kms.TryInit()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
          }
          hdmi_ready.store(1);
          last_ver = 0;
        }
        cv::Mat composite;
        {
          std::unique_lock<std::mutex> lk(comp_slot.mu);
          comp_slot.cv.wait_for(lk, std::chrono::milliseconds(500), [&] {
            return comp_slot.version > last_ver || g_stop.load();
          });
          if (comp_slot.version <= last_ver) continue;
          last_ver = comp_slot.version;
          composite = comp_slot.img;
        }
        if (composite.empty()) continue;
        if (panel_tick++ % 30 == 0) build_caches();
        // canvas: [gauge col | video | metric panel] + 3 QR slots (panel bottom)
        const int W = kms.Width(), H = kms.Height();
        cv::Mat canvas(H, W, CV_8UC3, kPnlBg);
        const int gw = gauge_size.width;
        const int pw = panel_size.width;
        const int vw = W - gw - pw - 16;
        if (!panel_cache.empty() && !gauge_cache.empty()) {
          // gauges: left column, vertically centered
          cv::Mat groi = canvas(cv::Rect(8, (H - gauge_size.height) / 2, gw,
                                         gauge_size.height));
          gauge_cache.copyTo(groi);
          // video center
          const double scale = std::min((double)(H - 12) / composite.rows,
                                        (double)(vw - 8) / composite.cols);
          const int dw = (int)(composite.cols * scale + 0.5);
          const int dh = (int)(composite.rows * scale + 0.5);
          const int dx = gw + 8 + (vw - dw) / 2;
          const int dy = (H - dh) / 2;
          cv::Mat vroi = canvas(cv::Rect(dx, dy, dw, dh));
          cv::Mat rs;
          cv::resize(composite, rs, vroi.size(), 0, 0, cv::INTER_AREA);
          rs.copyTo(vroi);
          // metric panel right
          cv::Mat proi = canvas(cv::Rect(W - pw, 0, pw, H));
          panel_cache.copyTo(proi);
          // 3 QR slots: bottom of the right panel area
          const int qr_h = (int)(H * 0.52);
          DrawQrColumn(canvas, cv::Point(W - pw + 8, H - qr_h - 8), pw - 16, qr_h);
        } else {
          cv::Mat rs;
          cv::resize(composite, rs, cv::Size(W, H), 0, 0, cv::INTER_AREA);
          rs.copyTo(canvas);
        }
        kms.Present(canvas);
        hdmi_flips.store(kms.flips());
        if (!kms.Ready()) {
          hdmi_ready.store(0);
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
      }
      kms.Close();
    });

    std::thread hist_feeder([&] {
      while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        std::lock_guard<std::mutex> lk(panel_mu);
        panel_det_hist = det_roll.tail(48);
        panel_dep_hist = dep_roll.tail(48);
      }
    });
    hist_feeder.detach();
  }

  // ---- HTTP ------------------------------------------------------------------
  system_metrics::Sampler system_sampler;
  auto build_stats = [&]() -> std::string {
    const system_metrics::Metrics system_snapshot = system_sampler.Sample();
    DetResult tr = store.det();
    auto [davg, dmin, dmax, dp95] = det_roll.stats();
    auto [pavg, pmin, pmax, pp95] = dep_roll.stats();
    auto [track_avg, track_min, track_max, track_p95] = track_roll.stats();
    double now_s = ms_since(t_start) / 1000.0;
    // sustained rate: ignore the first 2 s (warmup skew)
    double det_span = std::max(now_s - 2.0, 1.0);
    double dep_span = det_span;
    double det_fps = det_count.load() / det_span;
    double dep_fps = dep_count.load() / dep_span;
    if (now_s < 8.0) {
      det_fps = det_count.load() / std::max(now_s, 1.0);
      dep_fps = dep_count.load() / std::max(now_s, 1.0);
    }
    std::map<int, int> hist = tr.cls_hist;
    auto arr = [](const std::vector<double>& v) {
      std::string s = "[";
      char b[16];
      for (double x : v) {
        std::snprintf(b, sizeof(b), "%.2f,", x);
        s += b;
      }
      if (s.size() > 1) s.pop_back();
      s += "]";
      return s;
    };
    long long freq_mhz = atoll(
        read_first_line("/sys/class/devfreq/28108000.bpu/cur_freq").c_str()) /
                         1000000;
    std::ostringstream o;
    o << std::fixed << std::setprecision(1)
      << "{\"uptime_s\":" << now_s << ",\"display_fps\":" << display_fps.load()
      << ",\"capture\":{\"frames\":" << cam_count.load()
      << ",\"fps\":" << (cam_count.load() / std::max(now_s, 1.0)) << "},";
    o << "\"detect\":{\"frames\":" << det_count.load() << ",\"fps\":" << det_fps
      << std::setprecision(2)
      << ",\"bpu_ms\":{\"avg\":" << davg << ",\"min\":" << dmin
      << ",\"max\":" << dmax << ",\"p95\":" << dp95
      << "},\"pre_ms\":" << det_pre_ms.load()
      << ",\"post_ms\":" << det_post_ms.load()
      << ",\"objects\":" << tr.object_count
      << std::setprecision(2)
      << ",\"score_thres\":" << args.score
      << ",\"nms_thres\":" << args.nms
      << ",\"priority\":" << args.prio_det
      << ",\"bpu_hist\":" << arr(det_roll.tail(48)) << ",\"classes\":[";
    bool first = true;
    for (auto& kv : hist) {
      std::string name = (size_t)kv.first < labels.size()
                             ? labels[kv.first]
                             : "cls" + std::to_string(kv.first);
      if (!first) o << ",";
      first = false;
      const cv::Scalar& col = kUltra[kv.first % 16];
      char hexc[10];
      std::snprintf(hexc, sizeof(hexc), "#%02X%02X%02X", (int)col[2],
                    (int)col[1], (int)col[0]);
      o << "{\"name\":\"" << name << "\",\"count\":" << kv.second
        << ",\"color\":\"" << hexc << "\"}";
    }
    o << "]},\"tracking\":{"
      << "\"active\":" << tr.tracker_stats.active << ","
      << "\"total\":" << tr.tracker_stats.total_created << ","
      << "\"lost\":" << tr.tracker_stats.lost << ","
      << "\"ms\":" << track_avg << ","
      << "\"track_thresh\":" << args.track_thresh << ","
      << "\"low_thresh\":" << args.track_low_thresh << ","
      << "\"match_thresh\":" << args.track_match_thresh << ","
      << "\"buffer\":" << args.track_buffer << "},";
    o << "\"depth\":{\"frames\":" << dep_count.load()
      << ",\"fps\":" << dep_fps << std::setprecision(2)
      << ",\"bpu_ms\":{\"avg\":" << pavg << ",\"min\":" << pmin
      << ",\"max\":" << pmax << ",\"p95\":" << pp95
      << "},\"pre_ms\":" << dep_pre_ms.load()
      << ",\"post_ms\":" << dep_post_ms.load()
      << ",\"priority\":" << args.prio_dep
      << ",\"profile\":\"" << dep.profile() << "\""
      << ",\"bpu_hist\":" << arr(dep_roll.tail(48));
    {
      DepResult g = store.dep();
      double gmin = 1, gmax = 0;
      for (float v : g.grid) {
        if (v < gmin) gmin = v;
        if (v > gmax) gmax = v;
      }
      o << ",\"grid\":{\"cols\":" << g.grid_cols
        << ",\"rows\":" << g.grid_rows
        << ",\"unit\":\"" << (args.depth_meters > 0 ? "m" : "raw")
        << "\",\"scale\":" << std::setprecision(3) << args.depth_meters;
      if (!g.grid.empty())
        o << std::setprecision(2) << ",\"nearest\":" << gmin
          << ",\"farthest\":" << gmax
          << ",\"center\":" << g.grid[g.grid.size() / 2];
      o << "}";
    }
    o << "},";
    o << "\"cpu\":{\"util_pct\":" << std::setprecision(1)
      << system_snapshot.cpu_util_pct << "},";
    o << "\"memory\":{\"util_pct\":" << std::setprecision(1)
      << system_snapshot.memory_util_pct << "},";
    o << "\"display\":{\"hdmi\":" << (args.hdmi ? (hdmi_ready.load() ? 1 : 0) : -1)
      << ",\"mode\":\"" << (kms.Ready() ? std::to_string(kms.Width()) + "x" + std::to_string(kms.Height()) : std::string("off")) << "\""
      << ",\"flips\":" << hdmi_flips.load() << "},";
    o << "\"bpu\":{\"util_pct\":" << std::setprecision(1) << bpu_load()
      << ",\"freq_mhz\":" << freq_mhz
      << ",\"cores\":"
      << read_first_line("/sys/devices/system/bpu/core_num")
      << ",\"load_ms_per_frame\":" << (davg + pavg) << "},";
    o << "\"models\":{\"detect\":{\"file\":\"" << basename_of(args.det_model)
      << "\",\"input\":\"640x640 NV12\",\"latency_bench_ms\":9.55},"
      << "\"depth\":{\"file\":\"" << basename_of(args.dep_model)
      << "\",\"input\":\"" << dep.input_size() << "x" << dep.input_size() << " "
      << (dep.profile() == "lite" ? "F32 lite" : "NV12")
      << "\",\"latency_bench_ms\":13.55}},";
    o << "\"system\":{\"soc\":\"" << hbUCPGetSocName() << "\",\"dnn\":\""
      << hbDNNGetVersion() << "\",\"ucp\":\"" << hbUCPGetVersion() << "\"}}";
    return o.str();
  };

  HttpServer http(args.port, &shared, build_stats);
  if (!http.start()) {
    std::fprintf(stderr, "[FATAL] cannot bind port %d\n", args.port);
    g_stop = true;
  } else {
    std::printf("[http  ] http://0.0.0.0:%d/  (MJPEG + stats panel)\n",
                args.port);
  }

  // ---- capture loop (main thread) -------------------------------------------
  auto cap_t0 = NOW();
  uint64_t cap_frames = 0;
  while (!g_stop.load()) {
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
      std::printf("[camera] stream ended\n");
      break;
    }
    if (frame.cols & 1)
      frame = frame.colRange(0, frame.cols - 1);
    if (frame.rows & 1)
      frame = frame.rowRange(0, frame.rows - 1);
    hub.publish(frame);
    cam_count.fetch_add(1);
    ++cap_frames;
    double dt = ms_since(cap_t0) / 1000.0;
    if (args.max_seconds > 0 && dt > args.max_seconds) break;
  }

  g_stop = true;
  cap.release();
  det_worker.join();
  dep_worker.join();
  composer.join();
  encoder.join();
  if (hdmi_thread.joinable()) hdmi_thread.join();
  http.stop();
  hbDNNRelease(packed);

  double elapsed = ms_since(t_start) / 1000.0;
  auto [davg, dmin, dmax, dp95] = det_roll.stats();
  auto [pavg, pmin, pmax, pp95] = dep_roll.stats();
  std::printf(
      "[summary] elapsed %.1fs | cam %llu (%.1ffps) | detect %llu "
      "(avg bpu %.2fms p95 %.2f) | depth %llu (avg bpu %.2fms p95 %.2f) | "
      "display %.1ffps | bpu util %.1f%%\n",
      elapsed, (unsigned long long)cam_count.load(),
      cam_count.load() / std::max(elapsed, 0.1),
      (unsigned long long)det_count.load(), davg, dp95,
      (unsigned long long)dep_count.load(), pavg, pp95, display_fps.load(),
      bpu_load());
  return 0;
}
