# Neural data contract

`neural_data.h5` starts at the pooled-response level. Raw-pixel masking and 3 x 3 pooling are upstream preprocessing and are not performed here.

Required datasets:

| Dataset | Shape | Meaning |
|---|---:|---|
| `responses` | `(n_stimuli, n_units)` | One response vector per stimulus |
| `image_ids` | `(n_stimuli,)` | Unique stimulus identifiers, 1-based by default |
| `unit_coords_zero_based` | `(n_units, 2)` | Representative `(row, column)` for each pooled unit |
| `pool_counts` | `(n_units,)` | Number of raw valid pixels assigned to each unit |
| `unit_pixel_offsets` | `(n_units + 1,)` | CSR offsets into `unit_pixel_indices` |
| `unit_pixel_indices` | `(n_valid_pixels,)` | Flattened, zero-based raw mask indices |
| `map_mask` | `(height, width)` | Boolean valid-pixel map |

Alignment is explicit: response row `k` corresponds to `image_ids[k]`. The image path is formed with `image_filename_pattern`, for example `nsd_1000_{image_id:05d}.jpg`. The image ID is preserved as stored; subtract one only when indexing an unrelated zero-based array.

Unit IDs are always zero-based response-column indices. Thus `unit_02539` means `responses[:, 2539]`, the 2540th column when described in one-based human language. `unit_coords_zero_based` is only a representative coordinate. Exact raw-pixel membership for unit `u` is:

```python
flat = unit_pixel_indices[unit_pixel_offsets[u]:unit_pixel_offsets[u + 1]]
rows, columns = numpy.unravel_index(flat, map_mask.shape)
```

The validator rejects duplicate image IDs, missing images, non-finite responses, invalid coordinates and inconsistent CSR membership.
