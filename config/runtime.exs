import Config

xla_target = System.get_env("XLA_TARGET", "cpu")

preferred_clients =
  cond do
    String.starts_with?(xla_target, "cuda") -> [:cuda, :host]
    xla_target == "rocm" -> [:rocm, :host]
    true -> [:host]
  end

cuda_target? = String.starts_with?(xla_target, "cuda")
rocm_target? = xla_target == "rocm"
gpu_target? = preferred_clients != [:host]

preallocate? =
  System.get_env("EXLA_PREALLOCATE", if(gpu_target?, do: "false", else: "true"))
  |> String.downcase()
  |> case do
    "1" -> true
    "true" -> true
    "yes" -> true
    _ -> false
  end

memory_fraction =
  System.get_env("EXLA_MEMORY_FRACTION", if(gpu_target?, do: "0.25", else: "0.9"))
  |> Float.parse()
  |> case do
    {value, _rest} -> value
    :error -> if(gpu_target?, do: 0.25, else: 0.9)
  end

if cuda_target? do
  fallback_flag = "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"
  xla_flags = System.get_env("XLA_FLAGS", "")

  unless String.contains?(xla_flags, fallback_flag) do
    System.put_env("XLA_FLAGS", String.trim("#{xla_flags} #{fallback_flag}"))
  end
end

if rocm_target? do
  # gfx1201 (RDNA4 / Radeon RX 9000 series) is not recognized by the Triton GEMM
  # autotuner bundled with current XLA ROCm builds: it reports
  # DEVICE_TYPE_INVALID and aborts with "could not compile any configs".
  # Disabling GPU autotuning routes GEMMs through the stable rocBLAS path.
  # Leave any explicit autotune level the operator supplied untouched.
  autotune_flag = "--xla_gpu_autotune_level=0"
  xla_flags = System.get_env("XLA_FLAGS", "")

  unless String.contains?(xla_flags, "xla_gpu_autotune_level") do
    System.put_env("XLA_FLAGS", String.trim("#{xla_flags} #{autotune_flag}"))
  end

  # MIOpen ships no prebuilt FindDb for gfx1201, so every convolution JITs and
  # benchmarks solvers on first use. Persist the chosen kernels (MIOPEN_FIND_MODE
  # = FAST) so restarts reuse them instead of re-running the search.
  System.put_env("MIOPEN_FIND_MODE", System.get_env("MIOPEN_FIND_MODE", "2"))

  miopen_db_path = System.get_env("MIOPEN_USER_DB_PATH", "/tmp/miopen_user_db")

  case File.mkdir_p(miopen_db_path) do
    :ok -> :ok
    {:error, reason} -> IO.warn("Failed to create MIOpen user DB path #{miopen_db_path}: #{inspect(reason)}")
  end

  System.put_env("MIOPEN_USER_DB_PATH", miopen_db_path)
end

config :exla, :clients,
  cuda: [
    platform: :cuda,
    preallocate: preallocate?,
    memory_fraction: memory_fraction
  ],
  rocm: [
    platform: :rocm,
    preallocate: preallocate?,
    memory_fraction: memory_fraction
  ],
  tpu: [platform: :tpu],
  host: [platform: :host]

config :exla, :preferred_clients, preferred_clients
