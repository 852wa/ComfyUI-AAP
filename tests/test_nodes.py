import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


# Load the custom node without requiring a ComfyUI installation/package name.
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aap_nodes", ROOT / "nodes.py")
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


def convert(image, **overrides):
    options = dict(invert_alpha="enable", midrange_cut="disable", cut_threshold=0.5,
                   gamma_correction=2.2, remove_black_threshold=0.15, force_grayscale="enable")
    options.update(overrides)
    return nodes.AdvancedAlphaProcessor().convert(image, **options)


def legacy_reference(image, invert, cut, grayscale, gamma):
    """Original NumPy formulas, independent of the new tensor implementation."""
    outputs = [[], []]
    for img in image.numpy():
        rgb = np.power(img, gamma)
        if grayscale == "enable":
            gray = np.clip(np.dot(rgb, [0.2126, 0.7152, 0.0722]), 0, 1)
            rgb = np.stack([gray] * 3, axis=-1)
        alpha = np.dot(img, [0.299, 0.587, 0.114])
        if invert == "enable":
            alpha = 1.0 - alpha
        if cut == "enable":
            alpha = np.where(alpha >= 0.5, 1.0, 0.0)
        removed = alpha.copy()
        removed[np.all(img < 0.15, axis=-1)] = 0
        for target, opacity in zip(outputs, (alpha, removed)):
            target.append(np.concatenate((np.power(rgb * opacity[..., None], 1 / gamma),
                                          opacity[..., None]), axis=-1))
    return [torch.from_numpy(np.array(output)).float() for output in outputs]


@pytest.mark.parametrize("invert,cut,grayscale,gamma", itertools.product(
    ("enable", "disable"), ("enable", "disable"), ("enable", "disable"), (1.0, 2.2, 3.0)))
def test_legacy_rgb_compatibility(invert, cut, grayscale, gamma):
    image = torch.rand((3, 9, 13, 3), generator=torch.Generator().manual_seed(18))
    actual = convert(image, invert_alpha=invert, midrange_cut=cut,
                     force_grayscale=grayscale, gamma_correction=gamma)
    expected = legacy_reference(image, invert, cut, grayscale, gamma)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("mode", ["legacy", "straight"])
@pytest.mark.parametrize("cut", ["enable", "disable"])
def test_existing_alpha_is_preserved_after_threshold(mode, cut):
    image = torch.tensor([[[[0.25, 0.25, 0.25, 0.0], [0.25, 0.25, 0.25, 0.4]]]])
    for output in convert(image, alpha_mode=mode, midrange_cut=cut):
        torch.testing.assert_close(output[0, 0, 0], torch.zeros(4))
        assert output[0, 0, 1, 3].item() == pytest.approx(0.4 if cut == "enable" else 0.3)


def test_straight_alpha_png_compositing(tmp_path):
    image = torch.full((1, 1, 1, 3), 0.5)
    straight, _ = convert(image, alpha_mode="straight", force_grayscale="disable")
    legacy, _ = convert(image, force_grayscale="disable")
    torch.testing.assert_close(straight[0, 0, 0], torch.full((4,), 0.5))
    assert legacy[0, 0, 0, 0] < straight[0, 0, 0, 0]
    path = tmp_path / "alpha.png"
    Image.fromarray((straight[0].numpy() * 255).round().astype(np.uint8)).save(path)
    with Image.open(path) as foreground:
        assert foreground.mode == "RGBA"
        composite = Image.alpha_composite(Image.new("RGBA", (1, 1), "white"), foreground)
        assert composite.getpixel((0, 0)) == (191, 191, 191, 255)


def test_black_removal_strict_boundary_and_output_independence():
    image = torch.tensor([[[[0.149, 0.149, 0.149], [0.15, 0.15, 0.15], [0.0, 0.0, 0.5]]]])
    original, removed = convert(image)
    assert original[0, 0, 0, 3] > 0
    torch.testing.assert_close(removed[0, 0, 0], torch.zeros(4))
    torch.testing.assert_close(original[..., 1:, :], removed[..., 1:, :])
    assert original.data_ptr() != removed.data_ptr()
    no_removal = convert(image, remove_black_threshold=0)
    torch.testing.assert_close(*no_removal)


@pytest.mark.parametrize("threshold,expected", [(0, [1, 1, 1]), (0.5, [1, 1, 0]), (1, [1, 0, 0])])
def test_cut_boundaries(threshold, expected):
    image = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 3, 1).expand(-1, -1, -1, 3)
    output, _ = convert(image, midrange_cut="enable", cut_threshold=threshold)
    torch.testing.assert_close(output[0, 0, :, 3], torch.tensor(expected).float())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_noncontiguous_gradient_inputs_are_supported_and_unchanged(dtype):
    image = torch.rand(2, 4, 6, 3).to(dtype).transpose(1, 2).requires_grad_()
    before = image.detach().clone()
    for output in convert(image):
        assert output.shape == (2, 6, 4, 4)
        assert output.dtype == torch.float32
        assert not output.requires_grad
        assert output.device == image.device
    torch.testing.assert_close(image.detach(), before)


def test_grayscale_channel_matches_rgb():
    gray = torch.rand(2, 3, 4, 1)
    for actual, expected in zip(convert(gray), convert(gray.expand(-1, -1, -1, 3))):
        torch.testing.assert_close(actual, expected)


def test_nonfinite_and_out_of_range_pixels_are_sanitized():
    image = torch.tensor([[[[float("nan"), float("inf"), -float("inf")], [-1.0, 2.0, 0.5]]]])
    sanitized = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 1.0, 0.5]]]])
    for actual, expected in zip(convert(image), convert(sanitized)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3)])
def test_transparency_mask_broadcast(shape):
    image = torch.full((3, 2, 3, 3), 0.5)
    mask = torch.full(shape, 0.25)
    original = mask.clone()
    for output in convert(image, mask=mask):
        torch.testing.assert_close(output[..., 3], torch.full((3, 2, 3), 0.375))
    torch.testing.assert_close(mask, original)


def test_mask_batch_and_rgba_multiply():
    image = torch.full((2, 2, 3, 4), 0.5)
    mask = torch.stack((torch.zeros(2, 3), torch.ones(2, 3)))
    for output in convert(image, mask=mask):
        torch.testing.assert_close(output[0, ..., 3], torch.full((2, 3), 0.25))
        torch.testing.assert_close(output[1], torch.zeros(2, 3, 4))


def test_load_image_empty_mask_resizes():
    image = torch.rand(2, 9, 13, 3)
    for actual, expected in zip(convert(image, mask=torch.zeros(1, 64, 64)), convert(image)):
        torch.testing.assert_close(actual, expected)


def test_mask_resize_interpolates_transparency():
    image = torch.full((1, 1, 3, 3), 0.5)
    output, _ = convert(image, mask=torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(output[0, 0, :, 3], torch.tensor([0.5, 0.25, 0.0]))


@pytest.mark.parametrize("image", [None, torch.zeros(3, 4, 3), torch.zeros(1, 2, 3, 2),
                                  torch.zeros(0, 2, 3, 3), torch.zeros(1, 0, 3, 3),
                                  torch.zeros(1, 2, 3, 3, dtype=torch.uint8)])
def test_invalid_images_have_clear_errors(image):
    with pytest.raises(ValueError, match="image"):
        convert(image)


@pytest.mark.parametrize("options,name", [
    ({"gamma_correction": 0}, "gamma_correction"),
    ({"gamma_correction": float("nan")}, "gamma_correction"),
    ({"gamma_correction": "2.2"}, "gamma_correction"),
    ({"cut_threshold": -0.1}, "cut_threshold"),
    ({"remove_black_threshold": 1.1}, "remove_black_threshold"),
    ({"invert_alpha": "yes"}, "invert_alpha"),
    ({"midrange_cut": True}, "midrange_cut"),
    ({"force_grayscale": None}, "force_grayscale"),
    ({"alpha_mode": "unknown"}, "alpha_mode"),
    ({"mask": torch.zeros(2, 3, 3)}, "mask"),
    ({"mask": torch.zeros(1, 2, 3, 1)}, "mask"),
    ({"mask": torch.zeros(1, 0, 3)}, "mask"),
    ({"mask": torch.zeros(2, 3, dtype=torch.uint8)}, "mask"),
])
def test_invalid_parameters_have_clear_errors(options, name):
    with pytest.raises(ValueError, match=name):
        convert(torch.zeros(1, 2, 3, 3), **options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["legacy", "straight"])
def test_cuda_matches_cpu_and_accepts_cpu_mask(dtype, mode):
    image = torch.rand(2, 9, 13, 4).to(dtype)
    mask = torch.rand(1, 5, 7)
    for actual, expected in zip(convert(image.cuda(), mask=mask, alpha_mode=mode),
                                convert(image, mask=mask, alpha_mode=mode)):
        assert actual.is_cuda
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-6, rtol=3e-6)


def test_existing_example_widget_order_and_output_contract():
    workflow = json.loads((ROOT / "example.json").read_text())
    node = next(n for n in workflow["nodes"] if n["type"] == "AdvancedAlphaProcessor")
    schema = nodes.AdvancedAlphaProcessor.INPUT_TYPES()
    names = [name for name in schema["required"] if name != "image"]
    options = dict(zip(names, node["widgets_values"]))
    outputs = convert(torch.rand(1, 3, 4, 3), **options)
    assert names == ["invert_alpha", "midrange_cut", "cut_threshold", "gamma_correction",
                     "remove_black_threshold", "force_grayscale"]
    assert len(outputs) == len(nodes.AdvancedAlphaProcessor.RETURN_TYPES) == 2
    assert nodes.NODE_CLASS_MAPPINGS[node["type"]] is nodes.AdvancedAlphaProcessor


def test_custom_node_package_registration():
    name = "aap_test_package"
    package_spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[name] = package
    try:
        package_spec.loader.exec_module(package)
        cls = package.NODE_CLASS_MAPPINGS["AdvancedAlphaProcessor"]
        output = getattr(cls(), cls.FUNCTION)(torch.zeros(1, 2, 3, 3),
            "enable", "disable", 0.5, 2.2, 0.15, "enable")
        assert output[0].shape == (1, 2, 3, 4)
        assert "AdvancedAlphaProcessor" in package.NODE_DISPLAY_NAME_MAPPINGS
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop(name + ".nodes", None)


def test_straight_example_connections_and_execution():
    workflow = json.loads((ROOT / "example_straight.json").read_text())
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for link_id, source, source_slot, target, target_slot, kind in workflow["links"]:
        assert link_id in by_id[source]["outputs"][source_slot]["links"]
        assert by_id[target]["inputs"][target_slot]["link"] == link_id
        assert by_id[source]["outputs"][source_slot]["type"] == kind
        assert by_id[target]["inputs"][target_slot]["type"] == kind
    processor = next(n for n in by_id.values() if n["type"] == "AdvancedAlphaProcessor")
    args = processor["widgets_values"]
    assert args[-1] == "straight"
    image = torch.full((1, 2, 3, 3), 0.5)
    mask = torch.full((1, 2, 3), 0.25)
    for output in nodes.AdvancedAlphaProcessor().convert(image, *args, mask=mask):
        torch.testing.assert_close(output[..., :3], image)
        torch.testing.assert_close(output[..., 3], torch.full((1, 2, 3), 0.375))
