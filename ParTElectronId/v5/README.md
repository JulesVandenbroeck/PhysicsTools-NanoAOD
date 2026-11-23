# Electron ParT `full1` (v5aux3), epoch 39: ONNX with raw output

This folder has the electron model `full1` with 2 outputs per electron. The weights are the same as in the standard
export `epoch39/model.onnx`. The first output is the same as in the standard export. The second output is new.

Exported on 2026-11-02. This export was not tested in CMSSW.

## Files

| File | Content | SHA-256 |
|---|---|---|
| `model.onnx` | Model, opset 15, 22 MB | `f83a2f0563da57b4538d4e42118b7284b8e26a7929df46a0fac322442272c52d` |
| `preprocess.json` | Input names, order, scaling and lengths | `fb57c55abc16efad993d6832677fd62357a11bcd4b0694b80f34c9522c1df1dd` |

## Outputs

Each output has the shape (N, 16). N is the number of electrons. The data type is float32.

| Index | `output` | `raw` |
|---|---|---|
| 0 | P(prompt) | logit prompt |
| 1 | P(tau) | logit tau |
| 2 | P(b) | logit b |
| 3 | P(c) | logit c |
| 4 | P(conversion) | logit conversion |
| 5 | P(fake) | logit fake |
| 6 | P(charge flip) | logit charge flip |
| 7, 8, 9 | dressed q16, q50, q84 | dressed quantile values |
| 10, 11, 12 | bare q16, q50, q84 | bare quantile values |
| 13, 14, 15 | pre-FSR q16, q50, q84 | pre-FSR quantile values |

### pT targets

Each quantile in `output` estimates log(gen pT / `Lepton_pt`). To get a corrected pT, use `Lepton_pt * exp(q50)`.

| Target | Gen pT branch |
|---|---|
| dressed | `Lepton_genDressedPt` |
| bare (after FSR) | `Lepton_genPt` |
| pre-FSR | `Lepton_genPreFSRPt` |

### Relation between `raw` and `output`

`raw` is the output of the last linear layer, `Linear(192, 16)`. The steps below change `raw` into `output`:

1. Indices 0-5: softmax over the 6 values.
2. Index 6: sigmoid.
3. Each group of 3 pT values:
   - Sort the 3 values in ascending order.
   - Multiply by 0.04.
   - Limit q50 to the range -log(3) to +log(3).
   - Make sure that q16 <= q50 <= q84, with q16 = min(q16, q50) and q84 = max(q84, q50).

Do not read the 3 raw pT values in a group as q16, q50 and q84 before you sort them. The order is not guaranteed.

### What `raw` does not give

`output` is calculated from `raw`, and `raw` holds little more information. Softmax removes only a common offset of
the 6 logits. The sort and the limit change only the pT values. `raw` is not a general representation of the electron.
That representation is the 192-dimensional vector before the linear layer, and this file does not export it.

## Inputs

The model has 33 inputs: a features, a vectors and a mask input for each of the 11 groups `cpf`, `npf`, `lt`, `sv`,
`cl`, `rh`, `rho`, `hh`, `es`, `esc` and `lepton`. The shape of each input is (N, number of variables, number of
objects). The object axis is dynamic, except for `lepton` (fixed at 1).

Use `preprocess.json` for each input:
- `input_names`: the order of the inputs.
- `var_names`: the order of the variables in an input.
- `var_length`: the maximum number of objects that was used for the tests.
- `var_infos`: scaling, for each variable. The value is `(x - median) * norm_factor`, then limited to
  `lower_bound` and `upper_bound`. `replace_inf_value` replaces infinite values. `pad` fills empty positions.

## Checks

Check on 2000 electrons from 8 files of the DY test list `dy2.txt`, with ONNX Runtime on CPU:

| Comparison | Max absolute difference |
|---|---|
| `output` to `output` of `epoch39/model.onnx` | 0 |
| `output` to torch | 7.6e-5 |
| `raw` to torch | 2.5e-3 (raw values go up to about 31) |
| Steps above applied to `raw`, to `output` | 2.4e-7 |

`preprocess.json` is byte-identical to the file of the standard export.

## How to make this file again

```
python3 weaver/weaver/train.py --network-config networks/ParT_electron_v5aux3raw.py \
    --data-config data/electron_ParT_v5aux3_pq.yaml \
    --model-prefix model/electron_ParT_v5aux3.ddp4-bs2048-lr7p00e-04.nepoch40.parts2024pq.full1/net_epoch-39_state.pt \
    --export-onnx <output folder>/model.onnx
```

Source: `weaver-core-dev`, branch `dev-lepton`. The changes for this export are not committed (`raw_output` in
`networks/ParticleTransformerElectronV5.py`, new file `networks/ParT_electron_v5aux3raw.py`).

