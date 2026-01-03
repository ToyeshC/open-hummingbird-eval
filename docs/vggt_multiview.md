# VGGT Multi-View Loader

The MVImgNet integration now builds four-frame clips for VGGT while keeping the
existing Hummingbird evaluation pipeline intact.

## Data Sources

- **Angle-binned split** (`datasets/split_angles_mvimagenet`): drives the train /
  validation split and supplies the query frame plus its semantic mask.
- **Raw MVImgNet** (`datasets/mvimgnet`): provides the pool of available views
  for every object ID, **and** the COLMAP sparse reconstructions used to inject
  camera geometry (intrinsics, extrinsics, sparse points).

Both roots are joined via the `<class_id>/<object_id>` hierarchy. If the raw
folder is missing, the loader gracefully falls back to single-view mode.

## Sampling Strategy

1. Parse each angle-binned filename (`<object_id>_<idx>.jpg`) to recover the
   `object_id`.
2. Collect all raw views for that `object_id`.
3. Uniformly select three support frames across the list (0%, 33%, 66%, 100%),
   padding with the last available view if fewer than three exist.
4. Combine the angle-bin query frame and the sampled support frames into a
   clip of length four, preserving the query frame at index 0.

The helper `sample_uniform_views` performs the uniform selection and padding.

## Returned Sample

Each dataset item now returns a dictionary:

- `views`: `(S, C, H, W)` tensor with the query frame first and support views
  following.
- `mask`: semantic mask tensor `(1, H, W)` aligned with the query frame
  (present when masks are available).
- `label`: integer class ID.
- `bin`: angle-bin string used for logging and analytics.
- `object_id`: original MVImgNet object identifier.
- `camera`: dictionaries of stacked tensors `(S, 3, 3)` intrinsics,
  `(S, 4, 4)` world→camera, and `(S, 4, 4)` camera→world transforms sourced from
  COLMAP.

Default `S=4`, but the sequence length remains configurable via
`--vggt-seq-len`.

## Transform Handling

The loader reuses the existing `CombTransforms` pipeline. For support frames,
the same transform object runs with a dummy mask to keep resize/crop behaviour
consistent. If no transform is supplied, the loader falls back to a simple
`ToTensor` conversion to preserve the expected tensor types.

## Compatibility

- Memory creation and evaluation paths accept the new dictionary batches and
  continue to operate on query-aligned masks.
- Other datasets remain untouched and still return the original tuple format,
  so their behaviour is unchanged.

## Geometry-aware Feature Extraction

`VGGTFeatureExtractor` now consumes the multi-view clip *and* the per-view camera
metadata. During the forward pass we:

1. Run the VGGT aggregator to obtain fused tokens.
2. Execute the geometry heads (camera pose, depth, 3D points) in inference mode.
3. Pool predicted depth and world coordinates over the patch grid of the query
   view and concatenate them to the token embedding.
4. Keep the raw predictions (pose encodings, depth / point confidences) in the
   `extras` dictionary returned to the evaluation loop for downstream logging or
   inspection.

If COLMAP metadata exists, transforms are transferred to the GPU alongside the
clip so that downstream modules can align predictions in world space.

This setup allows VGGT to consume a lightweight multi-view context while
maintaining feature extraction and evaluation parity with the rest of the
Hummingbird encoders.

