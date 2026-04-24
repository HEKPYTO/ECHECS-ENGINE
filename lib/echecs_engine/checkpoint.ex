defmodule EchecsEngine.Checkpoint do
  @moduledoc """
  Utilities for loading and saving model and training checkpoints.
  """

  alias Axon.Loop.State

  @format "echecs-engine-model-state"
  @model_schema_version 2
  @latest_path "models/echecs_engine_latest.axon"
  @production_path "models/echecs_engine_production.axon"
  @evaluator_path "models/echecs_engine_evaluator.axon"

  @spec latest_path() :: String.t()
  def latest_path, do: @latest_path

  @spec production_path() :: String.t()
  def production_path, do: @production_path

  @spec evaluator_path() :: String.t()
  def evaluator_path, do: @evaluator_path

  @spec default_model_paths() :: [String.t()]
  def default_model_paths, do: [@production_path, @latest_path]

  @spec default_evaluator_paths() :: [String.t()]
  def default_evaluator_paths, do: [@evaluator_path]

  @spec model_schema_version() :: pos_integer()
  def model_schema_version, do: @model_schema_version

  @spec metadata(map()) :: map()
  def metadata(overrides \\ %{}) do
    base = %{
      format: @format,
      model_schema_version: @model_schema_version,
      tensor_schema: "119-plane-v3",
      policy_schema: "8x8x73-v1",
      output_heads: ["policy", "wdl", "moves_left"],
      app: "ECHECS-ENGINE",
      dependency_versions: dependency_versions(),
      training_config: %{"source" => "unknown"},
      saved_at: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    Map.merge(base, overrides, fn
      :training_config, left, right when is_map(left) and is_map(right) -> Map.merge(left, right)
      _key, _left, right -> right
    end)
  end

  @spec load_model_state(String.t() | [String.t()]) :: {:ok, map()} | {:error, term()}
  def load_model_state(paths) when is_list(paths) do
    Enum.reduce_while(paths, {:error, :enoent}, fn path, _acc ->
      case load_model_state(path) do
        {:ok, model_state} -> {:halt, {:ok, model_state}}
        {:error, :enoent} -> {:cont, {:error, :enoent}}
        {:error, reason} -> {:halt, {:error, {path, reason}}}
      end
    end)
  end

  def load_model_state(path) when is_binary(path) do
    with {:ok, %{model_state: model_state}} <- load_model_checkpoint(path) do
      {:ok, model_state}
    end
  end

  @spec load_model_checkpoint(String.t()) ::
          {:ok, %{model_state: map(), metadata: map()}} | {:error, term()}
  def load_model_checkpoint(path) when is_binary(path) do
    with {:ok, binary} <- File.read(path) do
      decode_model_checkpoint(binary)
    end
  end

  @spec save_model_state!(String.t(), map(), map()) :: :ok
  def save_model_state!(path, model_state, extra_metadata \\ %{}) do
    path
    |> Path.dirname()
    |> File.mkdir_p!()

    checkpoint = %{
      metadata: Map.merge(metadata(), extra_metadata),
      model_state_binary: Nx.serialize(model_state)
    }

    File.write!(path, :erlang.term_to_binary(checkpoint))
  end

  @spec load_training_state(String.t()) :: {:ok, State.t()} | {:error, term()}
  def load_training_state(path) do
    with {:ok, %{state: state}} <- load_training_checkpoint(path) do
      {:ok, state}
    end
  end

  @spec load_training_checkpoint(String.t()) ::
          {:ok, %{state: State.t(), metadata: map()}} | {:error, term()}
  def load_training_checkpoint(path) do
    with {:ok, binary} <- File.read(path) do
      decode_training_checkpoint(binary)
    end
  end

  @spec save_training_state!(String.t(), State.t(), map()) :: :ok
  def save_training_state!(path, %State{} = state, training_config \\ %{}) do
    path
    |> Path.dirname()
    |> File.mkdir_p!()

    checkpoint = %{
      metadata: metadata(%{training_config: training_config}),
      training_state_binary: Axon.Loop.serialize_state(state)
    }

    File.write!(path, :erlang.term_to_binary(checkpoint))
  end

  @spec load_evaluator_state(String.t()) ::
          {:ok, %{artifact: map(), metadata: map()}} | {:error, term()}
  def load_evaluator_state(paths) when is_list(paths) do
    Enum.reduce_while(paths, {:error, :enoent}, fn path, _acc ->
      case load_evaluator_state(path) do
        {:ok, artifact} -> {:halt, {:ok, artifact}}
        {:error, :enoent} -> {:cont, {:error, :enoent}}
        {:error, reason} -> {:halt, {:error, {path, reason}}}
      end
    end)
  end

  def load_evaluator_state(path) when is_binary(path) do
    with {:ok, binary} <- File.read(path) do
      try do
        case :erlang.binary_to_term(binary) do
          %{metadata: metadata, evaluator_artifact: payload} ->
            {:ok, %{artifact: deserialize_term(payload), metadata: metadata}}

          _other ->
            {:error, :invalid_evaluator_checkpoint}
        end
      rescue
        error -> {:error, error}
      end
    end
  end

  @spec save_evaluator_state!(String.t(), map(), map()) :: :ok
  def save_evaluator_state!(path, artifact, training_config \\ %{}) when is_map(artifact) do
    path
    |> Path.dirname()
    |> File.mkdir_p!()

    checkpoint = %{
      metadata:
        metadata(%{
          training_config: training_config,
          artifact_type: "sparse_evaluator"
        }),
      evaluator_artifact: serialize_term(artifact)
    }

    File.write!(path, :erlang.term_to_binary(checkpoint))
  end

  defp decode_model_checkpoint(binary) do
    case decode_training_checkpoint(binary) do
      {:ok, %{state: %State{step_state: %{model_state: model_state}}, metadata: metadata}} ->
        {:ok, %{model_state: model_state, metadata: metadata}}

      {:error, _reason} ->
        try do
          case :erlang.binary_to_term(binary) do
            %{metadata: metadata, model_state_binary: model_state_binary} ->
              model_state = Nx.deserialize(model_state_binary)
              {:ok, %{model_state: model_state, metadata: metadata}}

            _other ->
              decode_legacy_model_state(binary)
          end
        rescue
          _error -> decode_legacy_model_state(binary)
        end
    end
  end

  defp dependency_versions do
    %{
      "nx" => app_vsn(:nx),
      "axon" => app_vsn(:axon),
      "exla" => app_vsn(:exla),
      "echecs_engine" => app_vsn(:echecs_engine)
    }
  end

  defp app_vsn(app) do
    app
    |> Application.spec(:vsn)
    |> to_string()
  rescue
    _error -> "unknown"
  end

  defp decode_legacy_model_state(binary) do
    try do
      model_state = Nx.deserialize(binary)
      {:ok, %{model_state: model_state, metadata: Map.put(metadata(), :legacy?, true)}}
    rescue
      error -> {:error, error}
    end
  end

  defp decode_training_state(binary) do
    try do
      {:ok, Axon.Loop.deserialize_state(binary)}
    rescue
      error -> {:error, error}
    end
  end

  defp decode_training_checkpoint(binary) do
    try do
      case :erlang.binary_to_term(binary) do
        %{metadata: metadata, training_state_binary: training_state_binary} ->
          with {:ok, state} <- decode_training_state(training_state_binary) do
            {:ok, %{state: state, metadata: metadata}}
          end

        _other ->
          decode_legacy_training_checkpoint(binary)
      end
    rescue
      _error -> decode_legacy_training_checkpoint(binary)
    end
  end

  defp decode_legacy_training_checkpoint(binary) do
    with {:ok, state} <- decode_training_state(binary) do
      {:ok, %{state: state, metadata: Map.put(metadata(), :legacy?, true)}}
    end
  end

  defp serialize_term(%Nx.Tensor{} = tensor),
    do: %{__nx_tensor__: true, data: Nx.serialize(tensor)}

  defp serialize_term(term) when is_list(term), do: Enum.map(term, &serialize_term/1)

  defp serialize_term(term) when is_map(term) do
    Map.new(term, fn {key, value} -> {key, serialize_term(value)} end)
  end

  defp serialize_term(term) when is_tuple(term) do
    term
    |> Tuple.to_list()
    |> Enum.map(&serialize_term/1)
    |> List.to_tuple()
  end

  defp serialize_term(term), do: term

  defp deserialize_term(%{__nx_tensor__: true, data: binary}), do: Nx.deserialize(binary)
  defp deserialize_term(term) when is_list(term), do: Enum.map(term, &deserialize_term/1)

  defp deserialize_term(term) when is_map(term) do
    Map.new(term, fn {key, value} -> {key, deserialize_term(value)} end)
  end

  defp deserialize_term(term) when is_tuple(term) do
    term
    |> Tuple.to_list()
    |> Enum.map(&deserialize_term/1)
    |> List.to_tuple()
  end

  defp deserialize_term(term), do: term
end
