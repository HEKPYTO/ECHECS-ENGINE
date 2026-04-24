defmodule EchecsEngine.UCITest do
  use ExUnit.Case, async: true

  alias EchecsEngine.UCI

  test "responds to uci and isready handshakes" do
    state = UCI.new()

    {state, uci_output} = UCI.handle_line(state, "uci")
    {_state, ready_output} = UCI.handle_line(state, "isready")

    assert "id name ECHECS-ENGINE" in uci_output
    assert "uciok" in uci_output
    assert ready_output == ["readyok"]
  end

  test "sets startpos with moves and asks engine for bestmove" do
    best_move = fn fen, opts ->
      assert fen == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
      assert length(opts[:history_games]) == 2
      {:ok, "g1f3"}
    end

    state = UCI.new(best_move: best_move)
    {state, []} = UCI.handle_line(state, "position startpos moves e2e4 e7e5")

    {state, output} = UCI.handle_line(state, "go movetime 10")

    assert output == []
    assert {_state, ["bestmove g1f3"]} = drain_until_bestmove(state)
  end

  test "passes parsed go budgets to the search callback" do
    parent = self()

    best_move = fn _fen, opts ->
      send(parent, {:opts, opts})
      {:ok, "e2e4"}
    end

    state = UCI.new(best_move: best_move)
    {state, []} = UCI.handle_line(state, "go movetime 25 nodes 9 depth 3")
    assert {_state, ["bestmove e2e4"]} = drain_until_bestmove(state)

    assert_receive {:opts, opts}
    assert opts[:movetime] == 25
    assert opts[:nodes] == 9
    assert opts[:depth] == 3
  end

  test "emits iterative search info lines before bestmove" do
    best_move = fn _fen, opts ->
      opts[:reporter].(%{depth: 1, nodes: 32, pv: ["e2e4"]})
      opts[:reporter].(%{depth: 2, nodes: 64, pv: ["e2e4", "e7e5"]})
      {:ok, "e2e4"}
    end

    state = UCI.new(best_move: best_move)
    {state, []} = UCI.handle_line(state, "go depth 2")
    {_state, output} = drain_until_bestmove(state)

    assert Enum.at(output, 0) =~ "info depth 1"
    assert Enum.at(output, 1) =~ "info depth 2"
    assert List.last(output) == "bestmove e2e4"
  end

  test "passes standard clock controls and derives a movetime budget" do
    parent = self()

    best_move = fn _fen, opts ->
      send(parent, {:opts, opts})
      {:ok, "e2e4"}
    end

    state = UCI.new(best_move: best_move)

    {state, []} =
      UCI.handle_line(state, "go wtime 60000 btime 45000 winc 1000 binc 500 movestogo 30")

    assert {_state, ["bestmove e2e4"]} = drain_until_bestmove(state)

    assert_receive {:opts, opts}
    assert opts[:wtime] == 60_000
    assert opts[:btime] == 45_000
    assert opts[:winc] == 1_000
    assert opts[:binc] == 500
    assert opts[:movestogo] == 30
    assert is_integer(opts[:movetime])
    assert opts[:movetime] > 0
  end

  test "accepts infinite and stop commands" do
    parent = self()

    best_move = fn _fen, opts ->
      send(parent, {:opts, opts})
      Process.sleep(:infinity)
      {:ok, "e2e4"}
    end

    state = UCI.new(best_move: best_move)
    {state, []} = UCI.handle_line(state, "go infinite")
    {_state, ["bestmove 0000"]} = UCI.handle_line(state, "stop")

    assert_receive {:opts, opts}
    assert opts[:infinite] == true
  end

  test "normal go searches return immediately and can be stopped" do
    parent = self()

    best_move = fn _fen, opts ->
      send(parent, {:opts, opts})
      Process.sleep(:infinity)
      {:ok, "e2e4"}
    end

    state = UCI.new(best_move: best_move)
    {state, []} = UCI.handle_line(state, "go movetime 1000")
    assert_receive {:opts, opts}
    assert opts[:movetime] == 1000

    assert {_state, ["bestmove 0000"]} = UCI.handle_line(state, "stop")
  end

  test "quit cleans up active search task" do
    best_move = fn _fen, _opts ->
      Process.sleep(:infinity)
      {:ok, "e2e4"}
    end

    state = UCI.new(best_move: best_move)
    {state, []} = UCI.handle_line(state, "go infinite")
    {state, []} = UCI.handle_line(state, "quit")

    assert state.quit?
    assert state.search_task == nil
  end

  test "sets arbitrary FEN and applies moves" do
    best_move = fn fen, _opts ->
      assert fen == "8/8/8/8/4P3/8/8/4K2k b - e3 0 1"
      {:ok, "h1h2"}
    end

    state = UCI.new(best_move: best_move)
    {state, []} = UCI.handle_line(state, "position fen 8/8/8/8/8/8/4P3/4K2k w - - 0 1 moves e2e4")

    {state, output} = UCI.handle_line(state, "go")

    assert output == []
    assert {_state, ["bestmove h1h2"]} = drain_until_bestmove(state)
  end

  test "reports terminal positions as bestmove 0000" do
    best_move = fn _fen, _opts -> {:terminal, :stalemate} end

    state = UCI.new(best_move: best_move)
    {state, []} = UCI.handle_line(state, "go")
    {_state, output} = drain_until_bestmove(state)

    assert output == ["bestmove 0000"]
  end

  test "reports invalid position moves as info strings" do
    state = UCI.new()

    {_state, output} = UCI.handle_line(state, "position startpos moves e2e5")

    assert output == ["info string invalid position move e2e5"]
  end

  defp drain_until_bestmove(state, attempts \\ 20)

  defp drain_until_bestmove(state, 0), do: UCI.drain(state)

  defp drain_until_bestmove(state, attempts) do
    case UCI.drain(state) do
      {state, []} ->
        Process.sleep(10)
        drain_until_bestmove(state, attempts - 1)

      {state, output} when is_list(output) ->
        if Enum.any?(output, &String.starts_with?(&1, "bestmove ")) do
          {state, output}
        else
          Process.sleep(10)
          drain_until_bestmove(state, attempts - 1)
        end

      {state, output} ->
        {state, output}
    end
  end
end
