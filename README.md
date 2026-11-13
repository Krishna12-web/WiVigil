# Wi-Fi CSI sensing with ESP32

This project explores whether Wi-Fi channel state information (CSI) from two
ESP32 boards can distinguish an empty room, walking, and sitting. One board
sends Wi-Fi packets and the other receives them and streams CSI data to a
computer over USB serial. Python scripts collect and analyse the data.

The project is an experiment, not a reliable activity detector. Its current
recordings are small, and results vary a lot between sessions.

## What you need

- Two ESP32-S3-N16R8 boards (or compatible ESP32 boards supported by the
  Espressif CSI examples)
- A computer with Python installed
- Espressif's [esp-csi](https://github.com/espressif/esp-csi) examples and
  [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/)

## Set up Python

From the project directory, create and activate a virtual environment if you
want to keep the project packages separate, then install the dependencies:

```bash
python -m venv .venv
```

On Windows, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use:

```bash
source .venv/bin/activate
```

Then install the packages:

```bash
python -m pip install -r requirements.txt
```

## Prepare the ESP32 boards

1. Follow Espressif's ESP-IDF installation guide for your operating system.
2. In the `esp-csi` repository, find the `csi_send` and `csi_recv` example
   projects. Their locations can vary between versions.
3. Flash `csi_send` to one board and `csi_recv` to the other. Set both projects
   to use the same Wi-Fi channel.
4. Power the sender separately and connect the receiver to your computer.
   Place the boards a few metres apart and leave them in place while collecting
   a session.
5. Open the receiver's serial monitor and check that it prints lines beginning
   with `CSI_DATA,`. Note the baud rate configured in the example; the scripts
   below default to `921600`.

Close the serial monitor before running a Python script that uses the receiver
port. A serial port can generally only be used by one program at a time.

## Collect recordings

Each recording is saved as `data/<label>__<session>.npy`. The label describes
the activity; the session identifies a separate recording. For example:

```bash
python collect.py --port COM5 --label empty --seconds 90 --session s1
python collect.py --port COM5 --label walking --seconds 90 --session s1
python collect.py --port COM5 --label sitting --seconds 90 --session s1
```

Repeat for additional sessions, changing `s1` to `s2`, `s3`, and so on. On
Linux, the receiver port might look like `/dev/ttyUSB0`. Use `--baud` if your
receiver is configured for a different rate.

For the three labels, record an empty room, someone walking between the boards,
and someone sitting between them. Keep the hardware placement consistent and
write down any changes between sessions. The repository currently includes
recordings for all three labels across sessions `s1`, `s2`, and `s3`.

## Train and evaluate

Run the original train/evaluate script:

```bash
python train.py
```

It compares a Random Forest and an RBF SVM, holds out the alphabetically last
session for testing, prints per-class scores and a confusion matrix, then
refits the better model on all recordings and saves it as `model.joblib`.
Because that saved model has been refit on all the data, use the printed
held-out results—not predictions from the saved model—to report test
performance.

There is also a calibrated leave-one-session-out evaluation:

```bash
python train_v2.py
```

This reports results for each held-out session. It does not save a model.

With the recordings currently in the repository, `train.py` reports 0.48
accuracy and 0.424 macro F1 for its selected SVM on held-out session `s3`.
`train_v2.py` reports mean accuracy / macro F1 of 0.395 / 0.363 for the Random
Forest and 0.373 / 0.287 for the SVM across the three held-out sessions. These
results are weak, especially for sitting, and should not be presented as
reliable activity recognition.

## Try the live views

The browser dashboard can run without hardware using a generated signal:

```bash
python dashboard.py --demo
```

It opens at <http://127.0.0.1:8765>. To use a receiver instead, close other
programs using its serial port and run:

```bash
python dashboard.py --port COM5
```

The dashboard displays CSI amplitude, signal statistics, and motion after
empty-room calibration. It shows a model prediction if `model.joblib` exists.
Its breathing-rate estimate is experimental and should not be treated as a
measurement.

For a separate Matplotlib heatmap and classifier display, first run `train.py`
to create `model.joblib`, then start:

```bash
python live.py --port COM5
```

## Other scripts

- `csi_common.py` contains CSI parsing, cleanup, windowing, and feature
  extraction shared by the scripts.
- `diagnose.py` prints basic recording statistics and writes `diagnose.png`.
- `REPORT.md` contains the experiment details and per-session evaluation
  results.

## Limitations

CSI measurements depend on the room, board placement, people, and recording
conditions. The current model performs inconsistently across sessions and does
not reliably recognize sitting. The dashboard's motion indicator is based on
movement, so a person sitting completely still may not be detected as present.
The breathing estimate is a heuristic, not a medical measurement. This project
does not measure heart rate or body pose and is not intended for safety or
medical use.

Only collect data from people who have agreed to participate. This project
records CSI data, not images or audio.
