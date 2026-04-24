defmodule Mix.Tasks.Engine.TrainEvaluator do
  @moduledoc """
  Trains and exports a sparse evaluator artifact from supervised JSONL data.

  Usage:

      mix engine.train_evaluator path/to/train.jsonl
      mix engine.train_evaluator path/to/train.jsonl models/echecs_engine_evaluator.axon
  """

  use Mix.Task

  @shortdoc "Trains a sparse NNUE-style evaluator artifact"

  @impl Mix.Task
  def run(args) do
    Application.ensure_all_started(:echecs_engine)

    {flags, positional_args} = split_flags(args)
    {paths, output_path} = parse_args(positional_args)

    trainer_opts =
      [
        training_config: %{"dataset" => Enum.join(paths, ",")}
      ] ++ quantization_opts(flags)

    :ok = EchecsEngine.NNUE.Trainer.fit_and_save!(paths, output_path, trainer_opts)

    IO.puts("saved evaluator: #{output_path}")
  end

  defp parse_args([]),
    do: Mix.raise("usage: mix engine.train_evaluator train.jsonl [output.axon]")

  defp parse_args([dataset_path]), do: {[dataset_path], EchecsEngine.Checkpoint.evaluator_path()}

  defp parse_args([dataset_path, output_path]), do: {[dataset_path], output_path}

  defp parse_args(paths) do
    {output, datasets} = List.pop_at(paths, -1)
    {datasets, output}
  end

  defp split_flags(args) do
    Enum.split_with(args, &String.starts_with?(&1, "--"))
  end

  defp quantization_opts(flags) do
    if "--quantize" in flags do
      [quantize?: true]
    else
      []
    end
  end
end
