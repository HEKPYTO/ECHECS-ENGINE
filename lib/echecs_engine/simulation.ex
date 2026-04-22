defmodule EchecsEngine.Simulation do
  require Logger

  def run() do
    # Set EXLA as the default backend and compiler for the simulation
    Nx.global_default_backend(EXLA.Backend)
    Nx.Defn.global_default_options(compiler: EXLA)

    Logger.info("Setting up ECHECS-ENGINE Training Simulation on GPU...")

    # 1. Get the Model
    model = EchecsEngine.Model.build()

    # 2. Setup Loss Functions
    loss = fn
      %{policy: pred_p, value: pred_v}, %{policy: target_p, value: target_v} ->
        p_loss = Axon.Losses.categorical_cross_entropy(target_p, pred_p, reduction: :mean)
        v_loss = Axon.Losses.mean_squared_error(target_v, pred_v, reduction: :mean)
        Nx.add(p_loss, v_loss)
    end

    # 3. Setup Optimizer
    optimizer = :adam

    # 4. Generate Dummy Data
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

    # 5. Training Loop
    Logger.info("Starting training loop...")

    _trained_model_state =
      Axon.Loop.trainer(model, loss, optimizer)
      |> Axon.Loop.run(data, %{}, epochs: 3, compiler: EXLA)

    Logger.info("Training simulation completed successfully.")
  end
end
