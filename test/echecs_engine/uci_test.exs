defmodule EchecsEngine.UCITest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  alias EchecsEngine.UCI

  test "runs the UCI loop without Mix" do
    output = capture_io("uci\nisready\nquit\n", &UCI.run/0)

    assert output =~ "id name ECHECS-ENGINE"
    assert output =~ "uciok"
    assert output =~ "readyok"
  end

  test "responds to the UCI handshakes" do
    state = UCI.new()
    {state, output} = UCI.handle_line(state, "uci")
    assert "uciok" in output
    assert {_state, ["readyok"]} = UCI.handle_line(state, "isready")
  end

  test "stop returns the last completed principal variation" do
    state = UCI.new()
    {state, []} = UCI.handle_line(state, "position startpos")
    {state, []} = UCI.handle_line(state, "go infinite")
    {state, lines} = wait_for_info(state)
    assert Enum.any?(lines, &String.starts_with?(&1, "info depth "))
    task_pid = state.task.pid
    {state, stopped} = UCI.handle_line(state, "stop")
    assert Enum.any?(stopped, &String.starts_with?(&1, "bestmove "))
    assert Enum.count(stopped, &String.starts_with?(&1, "bestmove ")) == 1
    refute "bestmove 0000" in stopped
    assert is_nil(state.task)
    refute Process.alive?(task_pid)
    {_state, delayed} = UCI.drain(state)
    refute Enum.any?(delayed, &String.starts_with?(&1, "bestmove "))
  end

  test "many completed searches leave the application search supervisor at its baseline" do
    baseline = DynamicSupervisor.count_children(EchecsEngine.SearchSupervisor).active

    Enum.each(1..6, fn _ ->
      state = UCI.new()
      {state, []} = UCI.handle_line(state, "go depth 1")
      {state, _lines} = wait_for_completion(state)
      assert is_nil(state.task)
    end)

    assert DynamicSupervisor.count_children(EchecsEngine.SearchSupervisor).active == baseline
  end

  test "quit tears down the known supervised search task" do
    state = UCI.new()
    {state, []} = UCI.handle_line(state, "go infinite")
    {state, lines} = wait_for_info(state)
    assert Enum.any?(lines, &String.starts_with?(&1, "info depth "))
    task_pid = state.task.pid

    {state, []} = UCI.handle_line(state, "quit")
    assert state.quit?
    assert is_nil(state.task)
    refute Process.alive?(task_pid)
  end

  test "rejects an invalid position move" do
    {_state, [line]} = UCI.new() |> UCI.handle_line("position startpos moves e2e5")
    assert line =~ "invalid position move"
  end

  test "an errored or killed asynchronous task still emits one terminal bestmove" do
    task = Task.Supervisor.async_nolink(EchecsEngine.SearchSupervisor, fn -> {:error, :boom} end)

    state = %{
      UCI.new()
      | task: task,
        id: make_ref(),
        stop_ref: :atomics.new(1, []),
        bestmove: "e2e4"
    }

    {state, lines} = wait_for_completion(state)
    assert lines == ["info string search error :boom", "bestmove e2e4"]
    assert state.emitted?
    assert is_nil(state.task)
    assert {_state, []} = UCI.drain(state)

    {state, []} = UCI.new() |> UCI.handle_line("go infinite")
    task_pid = state.task.pid
    Process.exit(task_pid, :kill)
    {state, lines} = wait_for_completion(state)

    assert Enum.count(lines, &String.starts_with?(&1, "bestmove ")) == 1
    assert Enum.any?(lines, &String.starts_with?(&1, "info string search error "))
    refute "bestmove 0000" in lines
    assert state.emitted?
    assert is_nil(state.task)
    refute Process.alive?(task_pid)
    assert {_state, []} = UCI.drain(state)
  end

  test "invalid go preserves the protocol state for the next command" do
    state = UCI.new()
    {state, ["info string invalid go"]} = UCI.handle_line(state, "go depth 2 junk")
    assert {_state, ["readyok"]} = UCI.handle_line(state, "isready")
  end

  test "accepts standard clock controls with zero increments" do
    {state, []} =
      UCI.new()
      |> UCI.handle_line("go wtime 100 btime 100 winc 0 binc 0 movestogo 1")

    assert state.task
    {state, lines} = UCI.handle_line(state, "stop")
    assert is_nil(state.task)
    assert Enum.count(lines, &String.starts_with?(&1, "bestmove ")) == 1
  end

  test "UCI retains repetition history instead of round-tripping its game through FEN" do
    {state, []} =
      UCI.new()
      |> UCI.handle_line("position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8")

    assert Echecs.Game.draw?(state.game)
    {state, []} = UCI.handle_line(state, "go depth 1")
    {_state, transcript} = collect_completion(state)
    assert transcript == ["bestmove 0000"]
  end

  test "terminal positions and real go mate scores use UCI mate distances" do
    {state, []} = UCI.new() |> UCI.handle_line("position fen 7k/6Q1/7K/8/8/8/8/8 b - - 0 1")
    {state, []} = UCI.handle_line(state, "go depth 1")
    {_state, terminal} = collect_completion(state)
    assert terminal == ["bestmove 0000"]

    assert Enum.any?(go_transcript("7k/5Q2/7K/8/8/8/8/8 w - - 0 1", 2), &(&1 =~ "score mate 1"))

    assert Enum.any?(go_transcript("6k1/8/7K/8/8/8/8/5Q2 w - - 2 2", 4), &(&1 =~ "score mate 2"))

    assert Enum.any?(
             go_transcript("6k1/8/5K2/4Q3/8/8/8/8 b - - 0 1", 6),
             &(&1 =~ "score mate -2")
           )
  end

  defp wait_for_info(state, attempts \\ 50)
  defp wait_for_info(state, 0), do: UCI.drain(state)

  defp wait_for_info(state, attempts) do
    case UCI.drain(state) do
      {state, []} ->
        Process.sleep(10)
        wait_for_info(state, attempts - 1)

      {state, lines} ->
        {state, lines}
    end
  end

  defp wait_for_completion(state, attempts \\ 100)
  defp wait_for_completion(state, 0), do: UCI.drain(state)

  defp wait_for_completion(state, attempts) do
    case UCI.drain(state) do
      {%{task: nil} = state, lines} ->
        {state, lines}

      {state, _lines} ->
        Process.sleep(5)
        wait_for_completion(state, attempts - 1)
    end
  end

  defp go_transcript(fen, depth) do
    {state, []} = UCI.new() |> UCI.handle_line("position fen #{fen}")
    {state, []} = UCI.handle_line(state, "go depth #{depth}")
    {_state, lines} = collect_completion(state)
    lines
  end

  defp collect_completion(state, lines \\ [], attempts \\ 200)
  defp collect_completion(state, lines, 0), do: {state, lines}

  defp collect_completion(state, lines, attempts) do
    {state, output} = UCI.drain(state)
    lines = lines ++ output

    if is_nil(state.task) do
      {state, lines}
    else
      Process.sleep(5)
      collect_completion(state, lines, attempts - 1)
    end
  end
end
