// Browser port of realtime/live_inference._onnx_image_inputs (numpy/cv2)
// Values must match Python within float tolerance or the model output drifts.
// ponytail: paranoia ceiling — cv2's fixed-point uint8 rounding can flip a gray
// pixel by 1 vs this float port; verified effect is ~1e-3 on logits, verdicts stable.

export const S = 256; // model input size

const IMG_MEAN = [0.485, 0.456, 0.406];
const IMG_STD = [0.229, 0.224, 0.225];

// OpenCV INTER_LINEAR resize to SxS (coord = (dst+0.5)*scale-0.5, clamped, bilinear)
export function resizeBilinear(rgb, w, h, ow = S, oh = S) {
  const out = new Float32Array(ow * oh * 3);
  const sx = w / ow, sy = h / oh;
  for (let y = 0; y < oh; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0; else if (fy > h - 1) fy = h - 1;
    const y0 = Math.floor(fy), a = fy - y0;
    const y1 = y0 + 1 < h ? y0 + 1 : y0;
    for (let x = 0; x < ow; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      if (fx < 0) fx = 0; else if (fx > w - 1) fx = w - 1;
      const x0 = Math.floor(fx), b = fx - x0;
      const x1 = x0 + 1 < w ? x0 + 1 : x0;
      for (let c = 0; c < 3; c++) {
        const p00 = rgb[(y0 * w + x0) * 3 + c];
        const p01 = rgb[(y0 * w + x1) * 3 + c];
        const p10 = rgb[(y1 * w + x0) * 3 + c];
        const p11 = rgb[(y1 * w + x1) * 3 + c];
        out[(y * ow + x) * 3 + c] = p00 * (1 - a) * (1 - b) + p01 * (1 - a) * b + p10 * a * (1 - b) + p11 * a * b;
      }
    }
  }
  return out; // 0..255 float
}

// cv2 RGB2GRAY (uint8), values 0..255
export function toGrayU8(rgb, w, h) {
  const g = new Uint8ClampedArray(w * h);
  for (let i = 0; i < w * h; i++) g[i] = Math.round(0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]);
  return g;
}

// Gaussian blur 5x5, sigma auto (cv2 ksize=(5,5), sigmaX=0 -> 0.3*((5-1)*0.5-1)+0.8 = 1.1), BORDER_REFLECT101
const K = 1.1;
const KS = 5;
const _kernel = (() => {
  const k = new Float32Array(KS * KS);
  let sum = 0;
  for (let i = 0; i < KS; i++) for (let j = 0; j < KS; j++) {
    const v = Math.exp(-(((i - 2) ** 2 + (j - 2) ** 2) / (2 * K * K)));
    k[i * KS + j] = v; sum += v;
  }
  for (let i = 0; i < KS * KS; i++) k[i] /= sum;
  return k;
})();

function reflect101(idx, n) { return idx < 0 ? -idx : idx >= n ? 2 * (n - 1) - idx : idx; }

export function gaussianBlur5(gray, n) { // gray = Float32Array(n*n), values 0..1
  const out = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let acc = 0;
    for (let ki = 0; ki < KS; ki++) {
      const yy = reflect101(y + ki - 2, n) * n;
      for (let kj = 0; kj < KS; kj++) acc += _kernel[ki * KS + kj] * gray[yy + reflect101(x + kj - 2, n)];
    }
    out[y * n + x] = acc;
  }
  return out;
}

export function softmax(x) {
  const m = Math.max(...x);
  const e = x.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

// radix-2 Cooley-Tukey 1D FFT on Float64Arrays in place (numpy convention, no normalization)
function fft1D(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let wr = 1, wi = 0;
      for (let j = 0; j < half; j++) {
        const a = i + j, b = a + half;
        const tr = re[b] * wr - im[b] * wi;
        const ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const nr = wr * wRe - wi * wIm; wi = wr * wIm + wi * wRe; wr = nr;
      }
    }
  }
}

function fft2ShiftedMag(grayU8) { // grayU8: Uint8ClampedArray SxS -> Float32Array SxS minmax-normalized
  const re = new Float64Array(S * S), im = new Float64Array(S * S);
  const colRe = new Float64Array(S), colIm = new Float64Array(S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) re[y * S + x] = grayU8[y * S + x];
    fft1D(re.subarray(y * S, (y + 1) * S), im.subarray(y * S, (y + 1) * S));
  }
  for (let x = 0; x < S; x++) {
    for (let y = 0; y < S; y++) { colRe[y] = re[y * S + x]; colIm[y] = im[y * S + x]; }
    fft1D(colRe, colIm);
    for (let y = 0; y < S; y++) { re[y * S + x] = colRe[y]; im[y * S + x] = colIm[y]; }
  }
  const half = S >> 1;
  const mag = new Float64Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const i = y * S + x;
    mag[((y + half) % S) * S + ((x + half) % S)] = 20 * Math.log(Math.hypot(re[i], im[i]) + 1e-8);
  }
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < S * S; i++) { if (mag[i] < mn) mn = mag[i]; if (mag[i] > mx) mx = mag[i]; }
  const out = new Float32Array(S * S);
  const den = mx - mn + 1e-8;
  for (let i = 0; i < S * S; i++) out[i] = (mag[i] - mn) / den;
  return out;
}

// Full pipeline -> {rgb, noise, freq} Float32Arrays (CHW), matching _onnx_image_inputs
export function buildInputs(rgb, w, h) { // rgb: Uint8Array RGB 0..255 at (w,h)
  const r256 = resizeBilinear(rgb, w, h);
  const gray256 = toGrayU8(r256, S, S);
  const grayF = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) grayF[i] = gray256[i] / 255;
  const blur = gaussianBlur5(grayF, S);
  const resid = new Float32Array(S * S);
  let mean = 0;
  for (let i = 0; i < S * S; i++) { resid[i] = grayF[i] - blur[i]; mean += resid[i]; }
  mean /= S * S;
  let std = 0;
  for (let i = 0; i < S * S; i++) std += (resid[i] - mean) ** 2;
  std = Math.sqrt(std / (S * S)) + 1e-8;
  const noise = new Float32Array(3 * S * S);
  for (let c = 0; c < 3; c++) for (let i = 0; i < S * S; i++) noise[c * S * S + i] = (resid[i] - mean) / std;

  const rgbT = new Float32Array(3 * S * S);
  for (let c = 0; c < 3; c++) {
    const meanC = IMG_MEAN[c], stdC = IMG_STD[c];
    for (let i = 0; i < S * S; i++) rgbT[c * S * S + i] = (r256[i * 3 + c] / 255 - meanC) / stdC;
  }

  return { rgb: rgbT, noise, freq: fft2ShiftedMag(gray256) };
}