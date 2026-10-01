# Legacy code

Code that is no longer part of the package. Nothing here is included by
`src/LMP.jl`, and it has not been updated as the package changed, so it will
not run as is. It is kept for reference.

- `layers/`: layer types that no current model uses, such as the attention
  layers, PTC/CPTC, siteweight, BlockMatrix and several normalisation variants.
  Each file holds the struct, forward pass, backward pass and update of one
  layer, as they were in the old `src/` files.
- `unused/`: functions and constants from the old `src/` files that no current
  code, script or test calls. Each file is named after the old file it came
  from.
- `notebooks/`: the notebooks that exercised the legacy layers.
