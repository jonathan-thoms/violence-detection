# Agent Instructions

> This file is mirrored across CLAUDE.md, AGENTS.md, and GEMINI.md so the same instructions load in any AI environment.

You operate within a 3-layer architecture tailored for building a real-time, low-latency Computer Vision and AI application. LLMs are probabilistic, but high-FPS screen capture and video processing require deterministic, highly optimized code. This system fixes that mismatch.

## The 3-Layer Architecture

**Layer 1: Directive (What to do)**
- Standard Operating Procedures (SOPs) written in Markdown, living in `directives/`
- Define the goals, inputs, tools/scripts to use, outputs, and edge cases for each distinct CV module (e.g., `directives/face_gatekeeper.md`, `directives/yolo_inference.md`).
- Natural language instructions detailing the logic flow and latency constraints.

**Layer 2: Orchestration (Decision making)**
- This is you. Your job: intelligent routing and system integration.
- Read directives, call execution tools in the right order, handle threading/multiprocessing constraints, handle errors (like cv2 window crashes), and update directives with learnings.
- You are the glue between intent and execution. For example, you don't hallucinate YOLO bounding box coordinates; you read `directives/yolo_inference.md`, determine the optimal frame resizing, and then write/run `execution/run_yolo.py`.

**Layer 3: Execution (Doing the work)**
- Deterministic Python scripts in `execution/`
- Heavy lifting utilizing `mss` (screen capture), `cv2` (image manipulation/blurring), `face_recognition` (dlib), and `ultralytics` (YOLOv8).
- Environment variables, absolute paths to weights (`.pt` files), and reference images are configured in `.env`.
- Reliable, thread-safe, and highly optimized for Frames Per Second (FPS).

**Why this works:** Processing video frames in real-time is computationally expensive. If you try to write one giant monolithic script, threading issues and memory leaks compound. The solution is to isolate complexities into discrete, testable Python modules (face detection, screen capture, YOLO inference) before orchestrating them together.

## Operating Principles

**1. Check for tools first**
Before writing a new script, check `execution/` per your directive. Only create new scripts if none exist. Prioritize reusing established OpenCV window management and frame-reading boilerplate.

**2. Self-anneal when things break**
- Read the error message and stack trace (especially crucial for `numpy` shape mismatches or PyTorch tensor errors).
- Fix the script and test it again.
- Update the directive with what you learned (e.g., color space conversions like BGR to RGB, GPU memory limits, inference timing).
- Example: You hit an FPS bottleneck -> you look into the `mss` capture loop -> find a way to resize frames before inference -> rewrite script -> test -> update directive.

**3. Update directives as you learn**
Directives are living documents. When you discover library constraints, better OpenCV approaches, or threading expectations, update the directive. But do not create or overwrite directives without asking unless explicitly told to.

## File Organization

**Deliverables vs Intermediates:**
- **Deliverables**: The final cohesive GUI application (CustomTkinter/PyQt) and the trained YOLO `.pt` weights.
- **Intermediates**: Test scripts, sample video datasets, and single-frame extraction outputs.

**Directory structure:**
- `.tmp/` - All intermediate files (test frames, temporary annotated outputs). Never commit, always regenerated.
- `execution/` - Core Python scripts (the deterministic tools).
- `directives/` - SOPs in Markdown (the instruction set).
- `models/` - Stores YOLO weights (`best.pt`, `yolov8n.pt`) and face encodings. Added to `.gitignore` if files are too large.
- `data/` - Target face reference images and sample test videos.
- `.env` - Environment variables (e.g., paths, threshold confidences).

**Key principle:** Keep the main event loop extremely lightweight. Heavy AI inference should be offloaded to separate threads or processed asynchronously where possible.