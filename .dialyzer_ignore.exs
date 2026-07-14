# Dialyzer ignore file
# Suppresses warnings that are false positives due to missing type exports
# from external dependencies (:echecs, :axon).
[
  # Echecs.Move.t/0 is not exported from the :echecs dep; the spec is correct
  # at the call sites but Dialyzer cannot verify it.
  ~r/Invalid type specification for function.*update_after_move/,

  # Cascade from Dialyzer not resolving :echecs dep types for load_evaluator_weights.
  # The function correctly handles all return shapes from load_evaluator_state/1.
  ~r/Function load_evaluator_weights\/1 has no local return/,
  ~r/The function call load_evaluator_weights will not succeed/,
  ~r/The function call load_evaluator_state will not succeed/,

  # File.read/1 returns {:ok, binary} | {:error, posix}; Dialyzer may widen the
  # success type to include charlist/string from the OTP docs representation.
  ~r/The pattern can never match the type :eof \| binary\(\) \| string\(\)/
]
