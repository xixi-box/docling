# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    OcrMode,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.models.stages.ocr import tesseract_utils
from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel
from docling.models.stages.ocr.tesseract_ocr_model import TesseractOcrModel


class _RenderCaptured(Exception):
    pass


@pytest.mark.parametrize("engine", ["cli", "api"])
@pytest.mark.parametrize(
    "size, requested_scale, limited",
    [
        ((7063, 10301), 3.0, True),
        ((200, 100), 3.0, False),
        ((200, 100), 0.5, False),
        ((5000, 5000), 2.0, False),
    ],
    ids=["archival-scan", "small-page", "downscale", "at-budget"],
)
def test_tesseract_limits_render_before_allocation(
    tmp_path: Path,
    monkeypatch,
    engine: str,
    size,
    requested_scale: float,
    limited: bool,
) -> None:
    source = tmp_path / "scan.png"
    # A one-bit fixture compresses well, but the real image backend decodes it
    # to RGB, just as it does for an archival scan without DPI metadata.
    Image.new("1", size, 1).save(source)
    in_doc = InputDocument(
        path_or_stream=source, format=InputFormat.IMAGE, backend=ImageDocumentBackend
    )
    backend = in_doc._backend.load_page(0)
    page = Page(page_no=0, size=backend.get_size())
    page._backend = backend
    conv_res = ConversionResult(input=in_doc)

    model_type, options_type = (
        (TesseractOcrCliModel, TesseractCliOcrOptions)
        if engine == "cli"
        else (TesseractOcrModel, TesseractOcrOptions)
    )
    # Stop at the renderer boundary: no engine installation or huge rendered
    # buffer is needed to check the scale that the OCR stage actually requests.
    model = model_type(
        enabled=False,
        artifacts_path=None,
        options=options_type(
            lang=["eng"], mode=OcrMode.FULL_PAGE, scale=requested_scale
        ),
        accelerator_options=AcceleratorOptions(),
    )
    model.enabled = True
    if engine == "api":
        model._reader = object()
        model._osd_reader = object()

    def capture_render(scale, cropbox):
        pixels = round(cropbox.width * scale) * round(cropbox.height * scale)
        # Allow one pixel of rounding per axis at the 100 MP budget.
        assert pixels <= 100_020_001, "OCR render exceeds the pixel budget"
        if limited:
            assert scale < requested_scale
        else:
            assert scale == requested_scale
        raise _RenderCaptured

    monkeypatch.setattr(backend, "get_page_image", capture_render)
    try:
        with pytest.raises(_RenderCaptured):
            list(model(conv_res, [page]))
        assert model.scale == requested_scale
    finally:
        if engine == "api":
            model._reader = None
        backend.unload()
        in_doc._backend.unload()


class _Reader:
    """Engine boundary that returns one known OCR box in rendered pixels."""

    def End(self):
        pass

    def SetImage(self, image):
        pass

    def GetComponentImages(self, level, text_only):
        return [(None, {"x": 20, "y": 20, "w": 60, "h": 40}, 0, 0)]

    def SetRectangle(self, left, top, width, height):
        pass

    def GetUTF8Text(self):
        return "Archive"

    def MeanTextConf(self):
        return 90


@pytest.mark.parametrize("engine", ["cli", "api"])
@pytest.mark.parametrize("orientation", [0, 90])
def test_limited_scale_controls_coordinates_and_dpi(
    tmp_path: Path, monkeypatch, engine: str, orientation: int
) -> None:
    # Exercise the whole OCR stage with small buffers. Only the engine I/O is
    # replaced; region selection, rendering, rotation and cell merging are real.
    monkeypatch.setattr(tesseract_utils, "_TESSERACT_MAX_PIXELS", 20_000, raising=False)
    model_type, options_type = (
        (TesseractOcrCliModel, TesseractCliOcrOptions)
        if engine == "cli"
        else (TesseractOcrModel, TesseractOcrOptions)
    )
    model = model_type(
        enabled=False,
        artifacts_path=None,
        options=options_type(lang=["eng"], mode=OcrMode.FULL_PAGE),
        accelerator_options=AcceleratorOptions(),
    )
    model.enabled = True
    if engine == "api":
        model._reader = _Reader()
        model._osd_reader = SimpleNamespace(
            SetImage=lambda image: None,
            DetectOrientationScript=lambda: {"orient_deg": str(orientation)},
        )
        model._reader_RIL = SimpleNamespace(TEXTLINE=2)
    else:
        monkeypatch.setattr(
            model,
            "_perform_osd",
            lambda filename: pd.DataFrame(
                {"key": ["Orientation in degrees"], "value": [str(orientation)]}
            ),
        )

        def recognize(filename, osd):
            with Image.open(filename) as rendered:
                assert rendered.size == ((200, 100) if orientation == 0 else (100, 200))
                assert rendered.info["dpi"] == pytest.approx(
                    (72 * effective_scale, 72 * effective_scale), abs=0.02
                )
            return pd.DataFrame(
                [
                    {
                        "text": "Archive",
                        "conf": 90,
                        "left": 20,
                        "top": 20,
                        "width": 60,
                        "height": 40,
                    }
                ]
            )

        monkeypatch.setattr(model, "_run_tesseract", recognize)

    # Reuse the model for two page sizes. Each page needs its own effective scale.
    for size, effective_scale in [((200, 100), 1.0), ((100, 50), 2.0)]:
        source = tmp_path / "page.png"
        Image.new("RGB", size, "white").save(source)
        in_doc = InputDocument(
            path_or_stream=source,
            format=InputFormat.IMAGE,
            backend=ImageDocumentBackend,
        )
        backend = in_doc._backend.load_page(0)
        page = Page(page_no=0, size=backend.get_size())
        page._backend = backend
        try:
            assert list(model(ConversionResult(input=in_doc), [page])) == [page]
            assert [cell.text for cell in page.cells] == ["Archive"]
            bbox = page.cells[0].rect.to_bounding_box()
            expected = (20, 20, 80, 60) if orientation == 0 else (140, 20, 180, 80)
            assert bbox.as_tuple() == pytest.approx(
                tuple(value / effective_scale for value in expected)
            )
            assert model.scale == 3.0
        finally:
            backend.unload()
            in_doc._backend.unload()
