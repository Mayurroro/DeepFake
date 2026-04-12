# Roadmap

## Phase 1: Environment & Dataset Prep
**Goal:** Verify GPU functionality and prepare the dataset pipeline.
**Completion criteria:** CUDA is confirmed available in PyTorch, and dataset paths are successfully mounted/linked for the training script without crushing memory.

## Phase 2: Retrain ImageDetector (GPU)
**Goal:** Run the training loop for the image modality with GPU acceleration.
**Completion criteria:** Training completes, loss decreases, and validation accuracy is logged and improved.

## Phase 3: Retrain AudioClassifier (GPU)
**Goal:** Run the training loop for the audio modality with GPU acceleration.
**Completion criteria:** Audio training completes, utilizing Mel-spectrograms properly, and new weights are saved.

## Phase 4: Integration & Verification
**Goal:** Integrate new weights into the `deepfake_detector` and `deepfake_live_detector` and verify inference works End-to-End.
**Completion criteria:** System launches, correctly loads the new GPU-trained models, and predicts over a test image/audio file without crashing.

## Phase 5: Create Docker image for deepfake_detector

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 4
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 5 to break down)

## Phase 6: Create Docker image for deepfake_live_detector

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 5
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 6 to break down)
