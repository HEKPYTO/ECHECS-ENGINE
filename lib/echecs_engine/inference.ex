defmodule EchecsEngine.Inference do
  @moduledoc false

  @spec run((Nx.Tensor.t() -> map()) | atom(), Nx.Tensor.t()) :: map()
  def run(inference, tensor) when is_function(inference, 1), do: inference.(tensor)
  def run(inference, tensor), do: Nx.Serving.batched_run(inference, tensor)

  @spec run_batch((Nx.Tensor.t() -> map()) | atom(), Nx.Tensor.t()) :: map()
  def run_batch(inference, tensor), do: run(inference, tensor)
end
