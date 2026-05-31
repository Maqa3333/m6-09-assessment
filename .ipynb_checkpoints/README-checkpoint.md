# m6-09-assessment — Cat Detection v2

## Image for leaderboard
docker pull maqa333/cat-detector:final
Image: maqa333/cat-detector:final
Student: Mahammad Sadigov

## Run

### info
docker run --rm maqa333/cat-detector:final info

### predict
docker run --rm \
  -v /absolute/path/to/images:/data/input:ro \
  -v /absolute/path/to/results:/data/output \
  maqa333/cat-detector:final predict

Output is written to /data/output/predictions.csv

## Dataset
Same cat detection dataset from m6-04-assessment.
Split: 70/15/15 train/val/test.

## Model
- Framework: YOLO26
- Variant: yolo26n
- Epochs: 30
- mAP@0.5: 0.8353
- mAP@0.5:0.95: 0.5394
