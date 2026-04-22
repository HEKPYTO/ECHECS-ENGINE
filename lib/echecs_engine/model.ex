defmodule EchecsEngine.Model do
  @moduledoc """
  Provides the AlphaZero-style ResNet model built with `Axon`.
  """

  @doc """
  Builds the neural network architecture.

  Constructs an initial convolutional layer followed by a series of 
  residual blocks, terminating in dual heads (policy and value).
  Returns an `Axon` container mapping `%{policy: policy_head, value: value_head}`.
  """
  def build do
    input = Axon.input("input", shape: {nil, 119, 8, 8})

    x =
      input
      |> Axon.conv(256, kernel_size: {3, 3}, padding: :same)
      |> Axon.batch_norm()
      |> Axon.relu()

    x =
      x
      |> residual_block()
      |> residual_block()

    policy_head =
      x
      |> Axon.conv(2, kernel_size: {1, 1}, padding: :same)
      |> Axon.batch_norm()
      |> Axon.relu()
      |> Axon.flatten()
      |> Axon.dense(4672)
      |> Axon.softmax()

    value_head =
      x
      |> Axon.conv(1, kernel_size: {1, 1}, padding: :same)
      |> Axon.batch_norm()
      |> Axon.relu()
      |> Axon.flatten()
      |> Axon.dense(256)
      |> Axon.relu()
      |> Axon.dense(1)
      |> Axon.tanh()

    Axon.container(%{policy: policy_head, value: value_head})
  end

  @doc false
  defp residual_block(input) do
    x =
      input
      |> Axon.conv(256, kernel_size: {3, 3}, padding: :same)
      |> Axon.batch_norm()
      |> Axon.relu()
      |> Axon.conv(256, kernel_size: {3, 3}, padding: :same)
      |> Axon.batch_norm()

    Axon.add(input, x)
    |> Axon.relu()
  end
end
