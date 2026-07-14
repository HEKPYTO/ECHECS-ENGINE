defmodule Mix.Tasks.Engine.Uci do
  @moduledoc """
  Runs ECHECS-ENGINE as a UCI-compatible process.

  Usage:

      mix engine.uci
  """

  use Mix.Task

  @shortdoc "Runs the engine over the UCI protocol"

  @impl Mix.Task
  def run(_args) do
    Application.ensure_all_started(:echecs_engine)
    parent = self()
    input_reader = spawn(fn -> read_input(parent) end)
    loop(EchecsEngine.UCI.new(), input_reader)
  end

  defp read_input(parent) do
    case IO.gets("") do
      :eof ->
        send(parent, :uci_eof)

      {:error, _reason} ->
        send(parent, :uci_eof)

      line when is_binary(line) ->
        send(parent, {:uci_input, line})
        read_input(parent)

      line ->
        send(parent, {:uci_input, to_string(line)})
        read_input(parent)
    end
  end

  defp loop(state, input_reader) do
    receive do
      {:uci_input, line} ->
        {state, output} = EchecsEngine.UCI.handle_line(state, line)
        print_output(output)

        if state.quit? do
          stop_input_reader(input_reader)
        else
          loop(state, input_reader)
        end

      :uci_eof ->
        {state, output} = EchecsEngine.UCI.handle_line(state, "quit")
        print_output(output)
        stop_input_reader(input_reader)
        state
    after
      10 ->
        {state, output} = EchecsEngine.UCI.drain(state)
        print_output(output)
        loop(state, input_reader)
    end
  end

  defp print_output(output), do: Enum.each(output, &IO.puts/1)

  defp stop_input_reader(input_reader) do
    if Process.alive?(input_reader), do: Process.exit(input_reader, :kill)
  end
end
