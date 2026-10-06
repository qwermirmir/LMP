# Names of the submodules LMP used to have. Scripts written before the
# restructuring call e.g. LMP.Layer_sequence._read_raw_gauge_to_gpu! or
# LMP.gpu.sync_W_to_gpu!; every such name now lives in LMP itself, so the old
# module names simply point at LMP. Remove before the public release.
const elementry_base = LMP
const gpu = LMP
const Layer_types = LMP
const activation = LMP
const Optimizer = LMP
const Layer_sequence = LMP
