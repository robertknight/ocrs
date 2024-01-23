# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

 - Updated rten to v0.3.1. This improves performance on Arm by ~30%.

 - Fix panic in layout analysis when average word spacing in a line is negative
   [#20](https://github.com/robertknight/ocrs/pull/20)

 - Added LICENSE files to repository (Apache 2, MIT)
   [#12](https://github.com/robertknight/ocrs/pull/12)

## [0.3.1] - 2024-01-03

 - Fix cache directory location on Windows [#9](https://github.com/robertknight/ocrs/pull/9)

 - Improved speed on ARM by ~16% [9224ac9](https://github.com/robertknight/ocrs/commit/9224ac9)

## [0.3.0] - 2024-01-02

 - Extract the ocrs project out of the [RTen](https://github.com/robertknight/rten)
   repository and into a standalone repo at https://github.com/robertknight/ocrs.

 - Improve the `--json` output format with extracted text and coordinates of
   the rotated bounding rect for each word and line (92f17fb).

## [0.2.1] - 2024-01-01

 - Update rten to fix incorrect output on non-x64 / wasm32 platforms

## [0.2.0] - 2024-01-01

 - Improve layout analysis (ce52b3a1, cefb6c3f). The longer term plan is to use
   machine learning for layout analysis, but these incremental tweaks address
   some of the most egregious errors.
 - Add `--version` flag to CLI (20055ee0)
 - Revise CLI flags for specifying output format (97c3a011). The output path
   is now specified with `-o`. Available formats are text (default), JSON
   (`--json`) or annotated PNG (`--png`).
 - Fixed slow OCR model downloads by changing hosting location
   (https://github.com/robertknight/rten/issues/22).

## [0.1.0] - 2023-12-31

Initial release.
