defmodule EchecsEngine.Simulation do
  @moduledoc """
  Provides the training simulation loop for the ECHECS-ENGINE.

  This module sets up the neural network, loss functions, optimizer,
  and dummy self-play data generators to validate the `Axon` and `EXLA`
  compilation and training pipeline natively.
  """

  require Logger

  @doc """
  Runs the training simulation.

  Ensures necessary applications are started, configures `EXLA`, compiles the
  ResNet model, and trains on uniform random batches representing valid chess states.
  Saves the compiled state to a checkpoint upon successful execution.
  """
  def run() do
    Application.ensure_all_started(:exla)
    Application.ensure_all_started(:nx)
    Application.ensure_all_started(:axon)

    Nx.global_default_backend(EXLA.Backend)
    Nx.Defn.global_default_options(compiler: EXLA)

    Logger.info("Setting up ECHECS-ENGINE Training Simulation on GPU...")

    model = EchecsEngine.Model.ViT.build()

    loss = fn %{policy: pred_p, value: pred_v}, %{policy: target_p, value: target_v} ->
      p_loss = Axon.Losses.categorical_cross_entropy(target_p, pred_p, reduction: :mean)
      v_loss = Axon.Losses.mean_squared_error(target_v, pred_v, reduction: :mean)
      Nx.add(p_loss, v_loss)
    end

    optimizer = :adam

    batch_size = 64
    num_batches = 50

    Logger.info(
      "Generating #{num_batches} batches of dummy self-play data (batch size #{batch_size})..."
    )

    data =
      Stream.repeatedly(fn ->
        key = Nx.Random.key(System.system_time())
        {inputs, key} = Nx.Random.uniform(key, shape: {batch_size, 119, 8, 8}, type: :f32)

        {target_policy, key} = Nx.Random.uniform(key, shape: {batch_size, 4672}, type: :f32)

        target_policy =
          Nx.divide(target_policy, Nx.sum(target_policy, axes: [-1], keep_axes: true))

        {target_value, _key} = Nx.Random.uniform(key, shape: {batch_size, 1}, type: :f32)
        target_value = Nx.subtract(Nx.multiply(target_value, 2.0), 1.0)

        {inputs, %{policy: target_policy, value: target_value}}
      end)
      |> Enum.take(num_batches)

    Logger.info("Starting training loop...")

    checkpoint_path = "models/echecs_engine_v1.ckpt"

    initial_state =
      if File.exists?(checkpoint_path) do
        Logger.info("Found existing checkpoint at #{checkpoint_path}. Loading weights to resume training...")
        :erlang.binary_to_term(File.read!(checkpoint_path))
      else
        Logger.info("No checkpoint found. Training from scratch...")
        %{}
      end

    trained_model_state =
      Axon.Loop.trainer(model, loss, optimizer)
      |> Axon.Loop.run(data, initial_state, epochs: 10, compiler: EXLA)

    Logger.info("Training simulation completed successfully.")

    File.mkdir_p!("models")
    serialized = :erlang.term_to_binary(trained_model_state)
    File.write!(checkpoint_path, serialized)

    Logger.info("Model checkpoint saved to #{checkpoint_path}")
  end
end
