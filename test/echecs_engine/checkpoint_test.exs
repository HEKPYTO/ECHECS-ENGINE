defmodule EchecsEngine.CheckpointTest do
  use ExUnit.Case, async: true

  alias Axon.Loop.State
  alias EchecsEngine.Checkpoint

  setup do
    tmp_dir =
      System.tmp_dir!()
      |> Path.join("echecs_engine_checkpoint_test_#{System.unique_integer([:positive])}")

    File.mkdir_p!(tmp_dir)

    on_exit(fn -> File.rm_rf!(tmp_dir) end)

    %{tmp_dir: tmp_dir}
  end

  test "load_model_state/1 reads a production model state file", %{tmp_dir: tmp_dir} do
    model_state = %{"layer" => %{"kernel" => Nx.tensor([[1.0, 2.0]])}}
    path = Path.join(tmp_dir, "production.axon")

    File.write!(path, Nx.serialize(model_state))

    assert {:ok, loaded_state} = Checkpoint.load_model_state(path)

    assert Nx.equal(loaded_state["layer"]["kernel"], model_state["layer"]["kernel"])
           |> Nx.all()
           |> Nx.to_number() == 1
  end

  test "save_model_state!/2 stores model metadata and load_model_checkpoint/1 returns it", %{
    tmp_dir: tmp_dir
  } do
    model_state = %{"layer" => %{"kernel" => Nx.tensor([[1.0, 2.0]])}}
    path = Path.join(tmp_dir, "production.axon")

    :ok = Checkpoint.save_model_state!(path, model_state)

    assert {:ok, %{model_state: loaded_state, metadata: metadata}} =
             Checkpoint.load_model_checkpoint(path)

    assert metadata.model_schema_version == Checkpoint.model_schema_version()
    assert metadata.tensor_schema == "119-plane-v3"
    assert metadata.output_heads == ["policy", "wdl", "moves_left"]
    assert is_binary(metadata.dependency_versions["axon"])
    assert metadata.training_config["source"] == "unknown"

    assert Nx.equal(loaded_state["layer"]["kernel"], model_state["layer"]["kernel"])
           |> Nx.all()
           |> Nx.to_number() == 1
  end

  test "load_training_state/1 restores loop checkpoint including optimizer state", %{
    tmp_dir: tmp_dir
  } do
    model_state = %{"layer" => %{"kernel" => Nx.tensor([[3.0, 4.0]])}}

    training_state = %State{
      step_state: %{
        model_state: model_state,
        optimizer_state: %{momentum: Nx.tensor([0.5])}
      },
      epoch: 2,
      iteration: 9,
      metrics: %{"loss" => Nx.tensor(1.25)}
    }

    path = Path.join(tmp_dir, "latest.axon")
    File.write!(path, Axon.Loop.serialize_state(training_state))

    assert {:ok, loaded_state} = Checkpoint.load_training_state(path)
    assert loaded_state.epoch == 2
    assert loaded_state.iteration == 9

    assert Nx.equal(
             loaded_state.step_state.model_state["layer"]["kernel"],
             model_state["layer"]["kernel"]
           )
           |> Nx.all()
           |> Nx.to_number() == 1

    assert Nx.equal(
             loaded_state.step_state.optimizer_state.momentum,
             Nx.tensor([0.5])
           )
           |> Nx.all()
           |> Nx.to_number() == 1
  end

  test "save_training_state!/3 persists training checkpoint metadata", %{tmp_dir: tmp_dir} do
    training_state = %State{
      step_state: %{model_state: %{"layer" => %{"kernel" => Nx.tensor([[7.0]])}}},
      epoch: 1,
      iteration: 2
    }

    path = Path.join(tmp_dir, "latest.axon")
    :ok = Checkpoint.save_training_state!(path, training_state, %{"source" => "training-run"})

    assert {:ok, %{state: loaded_state, metadata: metadata}} =
             Checkpoint.load_training_checkpoint(path)

    assert loaded_state.epoch == 1
    assert metadata.training_config["source"] == "training-run"
    assert metadata.saved_at
  end

  test "load_model_state/1 can extract model weights from a full training checkpoint", %{
    tmp_dir: tmp_dir
  } do
    model_state = %{"layer" => %{"kernel" => Nx.tensor([[5.0, 6.0]])}}

    training_state = %State{
      step_state: %{
        model_state: model_state,
        optimizer_state: %{momentum: Nx.tensor([0.25])}
      }
    }

    path = Path.join(tmp_dir, "latest.axon")
    :ok = Checkpoint.save_training_state!(path, training_state, %{"source" => "training-run"})

    assert {:ok, loaded_state} = Checkpoint.load_model_state(path)
    assert {:ok, %{metadata: metadata}} = Checkpoint.load_model_checkpoint(path)
    assert metadata.training_config["source"] == "training-run"

    assert Nx.equal(loaded_state["layer"]["kernel"], model_state["layer"]["kernel"])
           |> Nx.all()
           |> Nx.to_number() == 1
  end

  test "rejects an evaluator checkpoint without creating atoms from its payload", %{
    tmp_dir: tmp_dir
  } do
    path = Path.join(tmp_dir, "untrusted.axon")
    atom_name = "untrusted_checkpoint_#{System.unique_integer([:positive])}"
    File.write!(path, <<131, 119, byte_size(atom_name), atom_name::binary>>)

    assert {:error, _reason} = Checkpoint.load_evaluator_state(path)
    assert_raise ArgumentError, fn -> String.to_existing_atom(atom_name) end
  end

  test "load_model_state/1 rejects a checkpoint with an incompatible model schema version",
       %{tmp_dir: tmp_dir} do
    model_state = %{"layer" => %{"kernel" => Nx.tensor([[1.0]])}}
    path = Path.join(tmp_dir, "production.axon")

    incompatible_version = Checkpoint.model_schema_version() + 100

    metadata =
      Checkpoint.metadata()
      |> Map.put(:model_schema_version, incompatible_version)

    checkpoint = %{metadata: metadata, model_state_binary: Nx.serialize(model_state)}
    File.write!(path, :erlang.term_to_binary(checkpoint))

    assert {:error, {:incompatible_schema, loaded, expected}} =
             Checkpoint.load_model_state(path)

    assert loaded == incompatible_version
    assert expected == Checkpoint.model_schema_version()
  end
end
