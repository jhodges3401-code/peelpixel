# Third-party licenses (verified Oct 12, 2026)
| Component | Use | License | Verified at |
|---|---|---|---|
| BiRefNet_lite ONNX (`onnx-community/BiRefNet_lite-ONNX`) | segmentation model, downloaded at runtime from Hugging Face | **MIT** (HF model card `license: mit`) | huggingface.co/api/models/onnx-community/BiRefNet_lite-ONNX |
| BiRefNet original weights/code (ZhengPeng7/BiRefNet) | upstream of the ONNX export | **MIT**, © 2024 ZhengPeng | github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE |
| ormbg ONNX (`onnx-community/ormbg-ONNX`, base `schirrmacher/ormbg`) | CPU/WASM fallback model (uint8), runtime download | **Apache-2.0** (both model cards) | huggingface.co/api/models/onnx-community/ormbg-ONNX |
| @huggingface/transformers 3.8.1 | inference runtime (loaded from jsDelivr) | **Apache-2.0** (npm metadata) | npm view @huggingface/transformers license |
| onnxruntime-web (dependency of the above) | WASM/WebGPU backend | **MIT** | npm view onnxruntime-web license |
All MIT/Apache-2.0: commercial use permitted with attribution/notice. PeelPixel's own code is original.
Rejected: briaai/RMBG-1.4 (non-commercial license), ISNet ONNX (AGPL-3.0), @imgly/background-removal (AGPL).
Test image `sample.jpg` (Wikimedia "Cat03.jpg", GFDL 1.2, Fir0002) is for local testing only. **Do not deploy it** (excluded via test/).
