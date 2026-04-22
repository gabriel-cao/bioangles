# BioAngles v1.1

**Joint Angle Extractor from Video using MediaPipe**

BioAngles extracts frame-by-frame joint angles from video using MediaPipe Tasks PoseLandmarker. Outputs CSV with angles per frame and JSON with statistics, including evaluation against CDC reference ranges (women 20-44).

Authors: Gabriel Cao Di Marco and Daniela Cao Di Marco — CONICET, Buenos Aires, 2026. DOI: 10.5281/zenodo.19209295

## Features

- 12 joint angles per frame: shoulder, elbow, hip, knee, ankle, trunk, neck (bilateral)
- CDC Range of Motion evaluation (women 20-44)
- Single video, batch mode, configurable frame sampling
- Three model complexities: lite / full / heavy
- Original finding: hip hyperextension patterns in AI-generated locomotion not previously documented in literature

## Install

pip install mediapipe opencv-python numpy pandas

## Usage

python3 bioangles.py video.mp4
python3 bioangles.py video.mp4 -o results/ -n 3 -m 2
python3 bioangles.py --batch /path/to/videos/

## Authors

- Gabriel Cao Di Marco — CONICET
- Daniela Cao Di Marco — CONICET

## License

MIT
