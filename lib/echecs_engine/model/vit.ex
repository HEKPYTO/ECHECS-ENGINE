defmodule EchecsEngine.Model.ViT do
  @moduledoc """
  Implements a Vision Transformer (ViT) architecture for Chess.
  """

  @doc """
  Builds the Transformer-based neural network architecture.
  """
  def build do
    input = Axon.input("input", shape: {nil, 119, 8, 8})

    x =
      input
      |> Axon.transpose([0, 2, 3, 1])
      |> Axon.reshape({:batch, 64, 119})

    embed_dim = 256
    x = Axon.dense(x, embed_dim, name: "patch_embedding")

    pos_embeddings =
      Axon.param("pos_embeddings", fn _ -> {1, 64, embed_dim} end, initializer: :normal)

    x = Axon.layer(fn x, pe, _opts -> Nx.add(x, pe) end, [x, pos_embeddings])

    x =
      x
      |> transformer_encoder_block(embed_dim, 8)
      |> transformer_encoder_block(embed_dim, 8)
      |> transformer_encoder_block(embed_dim, 8)
      |> transformer_encoder_block(embed_dim, 8)

    value_head =
      x
      |> Axon.global_avg_pool(keep_axes: false)
      |> Axon.dense(256)
      |> Axon.mish()
      |> Axon.dense(1)
      |> Axon.tanh()

    policy_head =
      x
      |> Axon.reshape({:batch, 8, 8, embed_dim})
      |> Axon.conv(73, kernel_size: {1, 1}, padding: :same)
      |> Axon.batch_norm()
      |> Axon.mish()
      |> Axon.flatten()

    Axon.container(%{policy: policy_head, value: value_head})
  end

  @doc false
  defp transformer_encoder_block(x, embed_dim, num_heads) do
    ln_x = Axon.layer_norm(x)

    qkv = Axon.dense(ln_x, embed_dim * 3, name: "qkv_proj")

    attention_out =
      Axon.nx(qkv, fn t ->
        head_dim = div(embed_dim, num_heads)
        {batch, seq_len, _} = Nx.shape(t)

        t = Nx.reshape(t, {batch, seq_len, 3, num_heads, head_dim})

        q = t[[.., .., 0, .., ..]]
        k = t[[.., .., 1, .., ..]]
        v = t[[.., .., 2, .., ..]]

        q = Nx.transpose(q, axes: [0, 2, 1, 3])
        k = Nx.transpose(k, axes: [0, 2, 1, 3])
        v = Nx.transpose(v, axes: [0, 2, 1, 3])

        scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
        scores = Nx.divide(scores, Nx.sqrt(head_dim))

        probs = Nx.exp(scores)
        sums = Nx.sum(probs, axes: [-1], keep_axes: true)
        probs = Nx.divide(probs, sums)

        out = Nx.dot(probs, [3], [0, 1], v, [2], [0, 1])

        out
        |> Nx.transpose(axes: [0, 2, 1, 3])
        |> Nx.reshape({batch, seq_len, embed_dim})
      end)

    attention_out = Axon.dense(attention_out, embed_dim, name: "out_proj")

    x = Axon.add(x, attention_out)

    ln_x2 = Axon.layer_norm(x)

    gate =
      ln_x2
      |> Axon.dense(embed_dim * 4, name: "swiglu_gate")
      |> Axon.mish()

    up = Axon.dense(ln_x2, embed_dim * 4, name: "swiglu_up")

    ffn_out =
      Axon.multiply(gate, up)
      |> Axon.dense(embed_dim, name: "swiglu_down")

    Axon.add(x, ffn_out)
  end
end
