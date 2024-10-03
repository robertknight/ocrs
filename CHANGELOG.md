# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2024-10-03

- Added `allowed_chars` library option and `--allowed-chars` CLI option to
  filter the characters produced by text recognition
  (https://github.com/robertknight/ocrs/pull/119). Thanks @basic-bgnr.

- Improved error message if custom alphabet size does not match recognition
  model output (https://github.com/robertknight/ocrs/pull/126)

- Updated README to note the importance of building the project, or at least the
  rten dependencies, in release mode
  (https://github.com/robertknight/ocrs/pull/118). Thanks @ezkangaroo.

## [0.8.1] - 2024-08-01

- Added ability to customize the alphabet used by the recognition model
  (https://github.com/robertknight/ocrs/pull/100). Thanks @Phaired.

- Updated rten to v0.13.1. This enables running custom models in the V2
  [rten model format](https://github.com/robertknight/rten/blob/main/docs/rten-file-format.md)

## [0.8.0] - 2024-05-25

### Breaking changes

This release changes Ocrs's internal use of threads, which may affect consumers
that are using their own parallelism. Specifically Ocrs no longer uses the
global Rayon thread pool but instead a custom thread pool which is sized to
match the number of physical rather than logical cores. See
https://github.com/robertknight/ocrs/pull/79 for more details and information on
adapting.

### Changes

- Updated rten to v0.10.0. This improves performance when recognizing long lines
  of text (https://github.com/robertknight/ocrs/pull/79) and improves efficiency
  by setting the number of threads to match the number of physical cores.

- Errors that occur when running the text recognition model are now propagated
  to the caller instead of causing a panic (https://github.com/robertknight/ocrs/pull/77)

## [0.7.0] - 2024-05-16

### Breaking changes

The APIs for loading models and images have changed in this release to make them
more efficient and easier to use. See the updated
[hello_ocr](https://github.com/robertknight/ocrs/blob/main/ocrs/examples/hello_ocr.rs)
example.

### Changes

 - Updated rten to v0.9.0. This brings a simpler API for loading models from
   disk (`Model::load_file`) and improves performance
   (https://github.com/robertknight/ocrs/pull/76)

 - Updated image crate. This includes a much faster JPEG decoder
   (https://github.com/robertknight/ocrs/pull/58)

 - Re-designed the API for loading images to be easier to use and more
   efficient (https://github.com/robertknight/ocrs/pull/56).

## [0.6.0] - 2024-04-29

 - Updated rten to v0.8.0. This fixes a crash on x86-64 CPUs that don't support
   AVX2 instructions and includes several performance improvements
   [#53](https://github.com/robertknight/ocrs/pull/53).

 - Added `--text-mask` flag to CLI which saves a binarized version of the text
   probability mask as an image (https://github.com/robertknight/ocrs/pull/38)

 - Made it easier to run examples (https://github.com/robertknight/ocrs/pull/41)

## [0.5.0] - 2024-02-28

 - Improve recognition accuracy for long text lines, at the cost of longer
   inference times, by increasing max image width after preprocessing
   [#32](https://github.com/robertknight/ocrs/pull/32)

 - Added `--text-line-images` option to save previews of text lines after
   preprocessing. This is useful for debugging recognition accuracy issues
   [#29](https://github.com/robertknight/ocrs/pull/29),
   [#30](https://github.com/robertknight/ocrs/pull/30).

 - Added a note in the README about the importance of building ocrs (or at least
   the rten dependencies) in release mode
   [#28](https://github.com/robertknight/ocrs/pull/28)

 - Updated rten to 0.4.0. This includes optimizations for post-processing of
   the text segmentation mask [#23](https://github.com/robertknight/ocrs/pull/23).

## [0.4.0] - 2024-01-23

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
