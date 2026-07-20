#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import importlib.util
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_IMAGE_SIZE = 256
DEFAULT_OUT_CHANNELS = 384


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        help='Run directory, e.g. results/.../2606111530')
    parser.add_argument('-s', '--subdataset', default='all',
                        help='Sub-dataset name, or "all" to export every '
                             'sub-dataset under run_dir/trainings')
    parser.add_argument('--output_dir', default=None,
                        help='Defaults to run_dir/onnx')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--image_size', type=int, default=None,
                        help='Defaults to the value stored in normalization.pt')
    parser.add_argument('--out_channels', type=int, default=None,
                        help='Defaults to the value stored in normalization.pt')
    parser.add_argument('--verify', action='store_true',
                        help='Run ONNX Runtime verification if onnxruntime is installed')
    return parser.parse_args()


def torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def ensure_onnx_available():
    if importlib.util.find_spec('onnx') is None:
        raise ImportError(
            'The `onnx` package is required for export. Install it with '
            '`pip install onnx`, then run this script again.')


def list_subdatasets(run_dir):
    trainings_dir = run_dir / 'trainings'
    if not trainings_dir.exists():
        raise FileNotFoundError('Training directory not found: {}'.format(
            trainings_dir))

    subdatasets = [
        path.name for path in sorted(trainings_dir.iterdir())
        if path.is_dir()
    ]
    if not subdatasets:
        raise FileNotFoundError('No sub-dataset folders found in {}'.format(
            trainings_dir))
    return subdatasets


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            '{} not found. Re-run training with the current efficientad.py '
            'so normalization.pt is saved.'.format(path))
    return path


class EfficientADOnnxWrapper(nn.Module):
    def __init__(self, teacher, student, autoencoder, normalization,
                 image_size, out_channels):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.image_size = int(image_size)
        self.out_channels = int(out_channels)

        self.register_buffer(
            'teacher_mean', normalization['teacher_mean'].float())
        self.register_buffer(
            'teacher_std', normalization['teacher_std'].float())
        self.register_buffer(
            'q_st_start', normalization['q_st_start'].float())
        self.register_buffer(
            'q_st_end', normalization['q_st_end'].float())
        self.register_buffer(
            'q_ae_start', normalization['q_ae_start'].float())
        self.register_buffer(
            'q_ae_end', normalization['q_ae_end'].float())

    def forward(self, image):
        teacher_output = self.teacher(image)
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std
        student_output = self.student(image)
        autoencoder_output = self.autoencoder(image)

        map_st = torch.mean(
            (teacher_output - student_output[:, :self.out_channels]) ** 2,
            dim=1, keepdim=True)
        map_ae = torch.mean(
            (autoencoder_output - student_output[:, self.out_channels:]) ** 2,
            dim=1, keepdim=True)

        map_st = 0.1 * (map_st - self.q_st_start) / (
            self.q_st_end - self.q_st_start)
        map_ae = 0.1 * (map_ae - self.q_ae_start) / (
            self.q_ae_end - self.q_ae_start)
        map_combined = 0.5 * map_st + 0.5 * map_ae

        map_combined = self._resize_map(map_combined)
        map_st = self._resize_map(map_st)
        map_ae = self._resize_map(map_ae)
        return map_combined, map_st, map_ae

    def _resize_map(self, anomaly_map):
        anomaly_map = F.pad(anomaly_map, (4, 4, 4, 4))
        return F.interpolate(
            anomaly_map, size=(self.image_size, self.image_size),
            mode='bilinear')


def build_wrapper(training_dir, image_size_override=None,
                  out_channels_override=None):
    teacher = torch_load(require_file(training_dir / 'teacher_final.pth'))
    student = torch_load(require_file(training_dir / 'student_final.pth'))
    autoencoder = torch_load(require_file(
        training_dir / 'autoencoder_final.pth'))
    normalization = torch_load(require_file(training_dir / 'normalization.pt'))

    image_size = image_size_override or normalization.get(
        'image_size', DEFAULT_IMAGE_SIZE)
    out_channels = out_channels_override or normalization.get(
        'out_channels', DEFAULT_OUT_CHANNELS)

    wrapper = EfficientADOnnxWrapper(
        teacher=teacher.eval(),
        student=student.eval(),
        autoencoder=autoencoder.eval(),
        normalization=normalization,
        image_size=image_size,
        out_channels=out_channels)
    return wrapper.eval(), int(image_size)


def export_subdataset(run_dir, output_dir, subdataset, opset,
                      image_size_override=None, out_channels_override=None,
                      verify=False):
    ensure_onnx_available()
    training_dir = run_dir / 'trainings' / subdataset
    wrapper, image_size = build_wrapper(
        training_dir=training_dir,
        image_size_override=image_size_override,
        out_channels_override=out_channels_override)

    export_dir = output_dir / subdataset
    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / 'efficientad.onnx'
    dummy_input = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['map_combined', 'map_st', 'map_ae'])

    if verify:
        verify_onnx(output_path, wrapper, dummy_input)

    print('Exported {} -> {}'.format(subdataset, output_path))


def verify_onnx(output_path, wrapper, dummy_input):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            'onnxruntime is required for --verify. Install it with '
            '`pip install onnxruntime`.') from exc

    with torch.no_grad():
        torch_outputs = wrapper(dummy_input)

    session = ort.InferenceSession(
        str(output_path), providers=['CPUExecutionProvider'])
    onnx_outputs = session.run(None, {'image': dummy_input.numpy()})

    for index, (torch_output, onnx_output) in enumerate(
            zip(torch_outputs, onnx_outputs)):
        max_diff = abs(torch_output.numpy() - onnx_output).max()
        if max_diff > 1e-4:
            raise RuntimeError(
                'ONNX verification failed for output {}: max diff {}'.format(
                    index, max_diff))


def main():
    config = get_argparse()
    run_dir = Path(config.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError('Run directory not found: {}'.format(run_dir))

    output_dir = Path(config.output_dir) if config.output_dir else \
        run_dir / 'onnx'
    subdatasets = list_subdatasets(run_dir) if config.subdataset == 'all' else \
        [config.subdataset]

    for subdataset in subdatasets:
        export_subdataset(
            run_dir=run_dir,
            output_dir=output_dir,
            subdataset=subdataset,
            opset=config.opset,
            image_size_override=config.image_size,
            out_channels_override=config.out_channels,
            verify=config.verify)


if __name__ == '__main__':
    main()
