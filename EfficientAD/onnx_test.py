#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True,
                        help='Path to efficientad.onnx')
    parser.add_argument('--image', default=None,
                        help='Single image path')
    parser.add_argument('--image_dir', default=None,
                        help='Directory of images to run recursively')
    parser.add_argument('--output_dir', default=None,
                        help='Defaults to ONNX parent directory/onnx_test')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--provider', default='CPUExecutionProvider',
                        help='ONNX Runtime provider, e.g. CPUExecutionProvider')
    parser.add_argument('--alpha', type=float, default=0.55,
                        help='Heatmap opacity for overlay images')
    parser.add_argument('--min_percentile', type=float, default=1.0,
                        help='Lower percentile for heatmap visualization')
    parser.add_argument('--max_percentile', type=float, default=99.0,
                        help='Upper percentile for heatmap visualization')
    return parser.parse_args()


def ensure_onnxruntime_available():
    if importlib.util.find_spec('onnxruntime') is None:
        raise ImportError(
            'The `onnxruntime` package is required. Install it with '
            '`pip install onnxruntime`, then run this script again.')


def pil_resampling_bilinear():
    if hasattr(Image, 'Resampling'):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def collect_images(image=None, image_dir=None):
    if image is None and image_dir is None:
        raise ValueError('Provide either --image or --image_dir.')

    images = []
    if image is not None:
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError('Image not found: {}'.format(path))
        images.append((path, Path(path.name)))

    if image_dir is not None:
        root = Path(image_dir)
        if not root.exists():
            raise FileNotFoundError('Image directory not found: {}'.format(root))
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((path, path.relative_to(root)))

    if not images:
        raise FileNotFoundError('No images found.')
    return images


def preprocess_image(path, image_size):
    original = Image.open(path).convert('RGB')
    resized = original.resize(
        (image_size, image_size), pil_resampling_bilinear())
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))[None]
    return original, array.astype(np.float32)


def squeeze_map(output):
    anomaly_map = np.asarray(output, dtype=np.float32)
    anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map.ndim != 2:
        raise ValueError('Expected 2D anomaly map output, got {}'.format(
            anomaly_map.shape))
    return np.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_for_visualization(anomaly_map, min_percentile, max_percentile):
    vmin = np.percentile(anomaly_map, min_percentile)
    vmax = np.percentile(anomaly_map, max_percentile)
    if vmax <= vmin:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    normalized = (anomaly_map - vmin) / (vmax - vmin)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def jet_colormap(normalized):
    x = np.clip(normalized, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    heatmap = np.stack([red, green, blue], axis=-1)
    return (255.0 * heatmap).astype(np.uint8)


def add_label(image, text):
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    try:
        font = ImageFont.truetype('arial.ttf', 18)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    padding = 6
    draw.rectangle((0, 0, width + 2 * padding, height + 2 * padding),
                   fill=(0, 0, 0))
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return labeled


def make_compare(original, heatmap, overlay, score):
    panels = [
        add_label(original, 'original'),
        add_label(heatmap, 'heatmap'),
        add_label(overlay, 'overlay max={:.6f}'.format(score)),
    ]
    width, height = original.size
    canvas = Image.new('RGB', (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    return canvas


def safe_output_stem(relative_path):
    return relative_path.with_suffix('')


def run_image(session, image_path, relative_path, output_dir, config):
    original, input_tensor = preprocess_image(image_path, config.image_size)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(None, {input_name: input_tensor})
    output_by_name = dict(zip(output_names, outputs))

    map_combined = squeeze_map(output_by_name.get('map_combined', outputs[0]))
    map_st = squeeze_map(output_by_name.get('map_st', outputs[1]))
    map_ae = squeeze_map(output_by_name.get('map_ae', outputs[2]))
    score = float(np.max(map_combined))

    stem = safe_output_stem(relative_path)
    image_output_dir = output_dir / stem.parent
    image_output_dir.mkdir(parents=True, exist_ok=True)
    base = stem.name

    combined_path = image_output_dir / '{}_map_combined.tiff'.format(base)
    st_path = image_output_dir / '{}_map_st.tiff'.format(base)
    ae_path = image_output_dir / '{}_map_ae.tiff'.format(base)
    heatmap_path = image_output_dir / '{}_heatmap.png'.format(base)
    overlay_path = image_output_dir / '{}_overlay.png'.format(base)
    compare_path = image_output_dir / '{}_compare.png'.format(base)

    tifffile.imwrite(combined_path, map_combined)
    tifffile.imwrite(st_path, map_st)
    tifffile.imwrite(ae_path, map_ae)

    normalized = normalize_for_visualization(
        map_combined, config.min_percentile, config.max_percentile)
    heatmap = Image.fromarray(jet_colormap(normalized), mode='RGB')
    heatmap = heatmap.resize(original.size, pil_resampling_bilinear())
    overlay = Image.blend(original, heatmap, float(np.clip(
        config.alpha, 0.0, 1.0)))
    compare = make_compare(original, heatmap, overlay, score)

    heatmap.save(heatmap_path)
    overlay.save(overlay_path)
    compare.save(compare_path)

    return {
        'image': str(image_path),
        'score_max': '{:.8f}'.format(score),
        'score_mean': '{:.8f}'.format(float(np.mean(map_combined))),
        'map_combined': str(combined_path),
        'map_st': str(st_path),
        'map_ae': str(ae_path),
        'heatmap': str(heatmap_path),
        'overlay': str(overlay_path),
        'compare': str(compare_path),
    }


def write_scores(output_dir, rows):
    scores_path = output_dir / 'scores.csv'
    with open(scores_path, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = [
            'image', 'score_max', 'score_mean', 'map_combined', 'map_st',
            'map_ae', 'heatmap', 'overlay', 'compare'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return scores_path


def main():
    config = get_argparse()
    ensure_onnxruntime_available()

    import onnxruntime as ort

    onnx_path = Path(config.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError('ONNX model not found: {}'.format(onnx_path))

    output_dir = Path(config.output_dir) if config.output_dir else \
        onnx_path.parent / 'onnx_test'
    output_dir.mkdir(parents=True, exist_ok=True)

    providers = [config.provider]
    available = ort.get_available_providers()
    if config.provider not in available:
        raise ValueError('Provider {} not available. Available: {}'.format(
            config.provider, ', '.join(available)))

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    images = collect_images(config.image, config.image_dir)
    rows = []
    for image_path, relative_path in images:
        row = run_image(session, image_path, relative_path, output_dir, config)
        rows.append(row)
        print('{} score={}'.format(image_path, row['score_max']))

    scores_path = write_scores(output_dir, rows)
    print('Wrote {} ONNX test results to {}'.format(len(rows), output_dir))
    print('Wrote scores to {}'.format(scores_path))


if __name__ == '__main__':
    main()
