# Infraction Dashboard

A web-based viewer for analyzing driving infractions from CARLA evaluations.
## Quick Start

### 1. Install Flask

```bash
pip install flask
```

### 2. Run the Dashboard

```bash
python lead/webapp/app.py
```

### 3. Open in Browser

Navigate to `http://localhost:5000`

## Usage

1. **Click "Load Routes"** to scan the default output directory (`outputs/local_evaluation/`)
2. **Select a route** from the sidebar to view its infractions
3. **Click any infraction** in the list to jump to that timestamp in the video
4. Use **video controls** for playback speed and frame-by-frame navigation

## Keyboard Shortcuts

- `Space` - Play/Pause
- `←` - Back 1 second
- `→` - Forward 1 second
- `Shift + ←` - Back 5 seconds
- `Shift + →` - Forward 5 seconds

## Custom Output Directory

To use a different evaluation directory, enter the path in the header input field and click "Load Routes".

## Requirements

- Flask
- Browser with HTML5 video support
- Evaluation data with `infractions.json` and video files
