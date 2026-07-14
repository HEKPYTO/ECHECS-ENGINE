# Dialyzer ignore file
# Suppresses false positives from external dependencies (:echecs, :axon)
# that do not export all types used in their public API.
[
  # Echecs.Move.t/0 is not exported from the :echecs dep.
  ~r/Invalid type specification for function.*update_after_move/,

  # Cascade: Dialyzer cannot resolve :echecs dep return types fully.
  ~r/Function load_evaluator_weights\/1 has no local return/,
  ~r/The function call load_evaluator_weights will not succeed/,
  ~r/The function call load_evaluator_state will not succeed/
]
