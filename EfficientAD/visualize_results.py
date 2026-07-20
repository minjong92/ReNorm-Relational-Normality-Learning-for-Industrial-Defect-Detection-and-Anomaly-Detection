#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle')
    parser.add_argument('-o', '--output_dir', default='results')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='E:/0_dataset/mvtec_anomaly_detection')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='E:/0_dataset/mvtec_loco_anomaly_detection')
    parser.add_argument('--visualization_dir', default=None,
                        help='Defaults to output_dir/visualizations/...')
    parser.add_argument('--alpha', type=float, default=0.55,
                        help='Heatmap opacity for overlay images')
    parser.add_argument('--normalization', default='image',
                        choices=['image', 'dataset'],
                        help='Normalize maps per image or by dataset min/max')
    parser.add_argument('--min_percentile', type=float, default=1.0,
                        help='Per-image lower percentile for visualization')
    parser.add_argument('--max_percentile', type=float, default=99.0,
                        help='Per-image upper percentile for visualization')
    return parser.parse_args()


def dataset_root(config):
    if config.dataset == 'mvtec_ad':
        return Path(config.mvtec_ad_path)
    if config.dataset == 'mvtec_loco':
        return Path(config.mvtec_loco_path)
    raise ValueError('Unknown dataset: {}'.format(config.dataset))


def dataset_output_name(dataset_path):
    dataset_name = Path(dataset_path).name
    if not dataset_name:
        raise ValueError('Invalid dataset path: {}'.format(dataset_path))
    return dataset_name


def has_anomaly_maps(run_dir, config):
    return (run_dir / 'anomaly_maps' / config.subdataset / 'test').exists()


def resolve_run_dir(config):
    output_dir = Path(config.output_dir)
    source_root = dataset_root(config)
    dataset_dir = output_dir / dataset_output_name(source_root)

    if has_anomaly_maps(output_dir, config):
        return output_dir

    candidates = []
    search_roots = []
    if dataset_dir.exists():
        search_roots.append(dataset_dir)
    if output_dir.exists() and output_dir not in search_roots:
        search_roots.append(output_dir)

    for search_root in search_roots:
        for child in search_root.iterdir():
            if not child.is_dir():
                continue
            if len(child.name) not in (10, 13):
                continue
            if not child.name[:10].isdigit():
                continue
            if len(child.name) == 13 and (
                    child.name[10] != '_' or not child.name[11:].isdigit()):
                continue
            if has_anomaly_maps(child, config):
                candidates.append(child)

    if candidates:
        run_dir = sorted(candidates, key=lambda path: path.name)[-1]
        print('Using latest run directory: {}'.format(run_dir))
        return run_dir

    return output_dir


def anomaly_map_root(config, run_dir):
    return run_dir / 'anomaly_maps' / config.subdataset / 'test'


def visualization_root(config, run_dir):
    if config.visualization_dir is not None:
        return Path(config.visualization_dir)
    return run_dir / 'visualizations' / config.subdataset / 'test'


def find_source_image(root, subdataset, defect_class, stem):
    image_dir = root / subdataset / 'test' / defect_class
    for ext in IMAGE_EXTENSIONS:
        path = image_dir / '{}{}'.format(stem, ext)
        if path.exists():
            return path

    for path in sorted(image_dir.glob('{}.*'.format(stem))):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def load_map(path):
    anomaly_map = tifffile.imread(str(path))
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.mean(axis=-1)
    if anomaly_map.ndim != 2:
        raise ValueError('Expected 2D anomaly map, got {} from {}'.format(
            anomaly_map.shape, path))
    return np.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_map(anomaly_map, config, dataset_min=None, dataset_max=None):
    if config.normalization == 'dataset':
        vmin = dataset_min
        vmax = dataset_max
    else:
        vmin = np.percentile(anomaly_map, config.min_percentile)
        vmax = np.percentile(anomaly_map, config.max_percentile)

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


def make_side_by_side(original, heatmap, overlay, score):
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


def compute_dataset_min_max(map_files):
    values_min = []
    values_max = []
    for path in map_files:
        anomaly_map = load_map(path)
        values_min.append(float(anomaly_map.min()))
        values_max.append(float(anomaly_map.max()))
    return min(values_min), max(values_max)


def main():
    config = get_argparse()
    config.alpha = float(np.clip(config.alpha, 0.0, 1.0))

    run_dir = resolve_run_dir(config)
    maps_root = anomaly_map_root(config, run_dir)
    source_root = dataset_root(config)
    output_root = visualization_root(config, run_dir)

    if not maps_root.exists():
        raise FileNotFoundError('Anomaly map directory not found: {}'.format(
            maps_root))
    if not source_root.exists():
        raise FileNotFoundError('Dataset directory not found: {}'.format(
            source_root))

    map_files = sorted(
        path for path in maps_root.rglob('*')
        if path.is_file() and path.suffix.lower() in ('.tif', '.tiff')
    )
    if not map_files:
        raise FileNotFoundError('No .tif/.tiff anomaly maps found in {}'.format(
            maps_root))

    dataset_min = None
    dataset_max = None
    if config.normalization == 'dataset':
        dataset_min, dataset_max = compute_dataset_min_max(map_files)

    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    missing_sources = []

    for map_path in map_files:
        defect_class = map_path.parent.name
        stem = map_path.stem
        source_path = find_source_image(
            source_root, config.subdataset, defect_class, stem)
        if source_path is None:
            missing_sources.append(str(map_path))
            continue

        original = Image.open(source_path).convert('RGB')
        anomaly_map = load_map(map_path)
        normalized = normalize_map(
            anomaly_map, config, dataset_min=dataset_min,
            dataset_max=dataset_max)

        heatmap = Image.fromarray(jet_colormap(normalized), mode='RGB')
        heatmap = heatmap.resize(original.size, Image.Resampling.BILINEAR)
        overlay = Image.blend(original, heatmap, config.alpha)
        score = float(np.max(anomaly_map))
        side_by_side = make_side_by_side(original, heatmap, overlay, score)

        class_output_dir = output_root / defect_class
        class_output_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = class_output_dir / '{}_heatmap.png'.format(stem)
        overlay_path = class_output_dir / '{}_overlay.png'.format(stem)
        side_by_side_path = class_output_dir / '{}_compare.png'.format(stem)

        heatmap.save(heatmap_path)
        overlay.save(overlay_path)
        side_by_side.save(side_by_side_path)

        rows.append({
            'defect_class': defect_class,
            'image': source_path.name,
            'anomaly_map': str(map_path),
            'score_max': '{:.8f}'.format(score),
            'score_mean': '{:.8f}'.format(float(np.mean(anomaly_map))),
            'heatmap': str(heatmap_path),
            'overlay': str(overlay_path),
            'compare': str(side_by_side_path),
        })

    scores_path = output_root / 'scores.csv'
    with open(scores_path, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = [
            'defect_class', 'image', 'score_max', 'score_mean',
            'anomaly_map', 'heatmap', 'overlay', 'compare'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print('Wrote {} visualizations to {}'.format(len(rows), output_root))
    print('Wrote scores to {}'.format(scores_path))
    if missing_sources:
        print('Skipped {} maps because source images were missing'.format(
            len(missing_sources)))


if __name__ == '__main__':
    main()
