defmodule EchecsZero.Model do
  @moduledoc """
  AlphaZero-style ResNet model built with Axon.
  """

  @doc """
  Builds the neural network model consisting of an initial convolutional block,
  several residual blocks, and dual heads (policy and value).
  """
  def build do
    input = Axon.input("input", shape: {nil, 119, 8, 8})

    # Initial Convolutional Block
    x =
      input
      |> Axon.conv(256, kernel_size: {3, 3}, padding: :same)
      |> Axon.batch_norm()
      |> Axon.relu()

    # Residual Blocks (we'll do 2 blocks for minimal implementation)
    x =
      x
      |> residual_block()
      |> residual_block()

    # Policy Head
    policy_head =
      x
      |> Axon.conv(2, kernel_size: {1, 1}, padding: :same)
      |> Axon.batch_norm()
      |> Axon.relu()
      |> Axon.flatten()
      |> Axon.dense(4672)
      |> Axon.softmax()

    # Value Head
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

    # Return container
    Axon.container(%{policy: policy_head, value: value_head})
  end

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
