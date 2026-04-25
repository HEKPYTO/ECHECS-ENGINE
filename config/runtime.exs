import Config

xla_target = System.get_env("XLA_TARGET", "cpu")

preferred_clients =
  cond do
    String.starts_with?(xla_target, "cuda") -> [:cuda, :host]
    true -> [:host]
  end

cuda_target? = String.starts_with?(xla_target, "cuda")
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

config :exla, :clients,
  cuda: [
    platform: :cuda,
    preallocate: preallocate?,
    memory_fraction: memory_fraction
  ],
  tpu: [platform: :tpu],
  host: [platform: :host]

config :exla, :preferred_clients, preferred_clients
