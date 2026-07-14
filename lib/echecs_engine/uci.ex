defmodule EchecsEngine.UCI do
  @moduledoc """
  Minimal UCI protocol state machine for engine integration and benchmarking.
  """

  alias Echecs.Game

  defstruct game: Echecs.new_game(),
            history_games: [],
            best_move: &EchecsEngine.best_move/2,
            task_supervisor: EchecsEngine.SearchTaskSupervisor,
            search_task: nil,
            search_id: nil,
            quit?: false

  @type t :: %__MODULE__{
          game: Game.t(),
          history_games: [Game.t()],
          best_move: (String.t(), keyword() ->
                        {:ok, String.t()} | {:terminal, atom()} | {:error, term()}),
          task_supervisor: atom() | pid(),
          search_task: Task.t() | nil,
          search_id: reference() | nil,
          quit?: boolean()
        }

  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      game: Echecs.new_game(),
      history_games: [],
      best_move: Keyword.get(opts, :best_move, &EchecsEngine.best_move/2),
      task_supervisor: Keyword.get(opts, :task_supervisor, EchecsEngine.SearchTaskSupervisor)
    }
  end

  @spec handle_line(t(), String.t()) :: {t(), [String.t()]}
  def handle_line(%__MODULE__{} = state, line) do
    {state, pending_output} = drain(state)

    {state, command_output} =
      line
      |> String.trim()
      |> String.split(~r/\s+/, trim: true)
      |> handle_tokens(state)

    {state, pending_output ++ command_output}
  end

  @spec drain(t()) :: {t(), [String.t()]}
  def drain(%__MODULE__{} = state), do: drain_messages(state, [])

  defp handle_tokens(["uci"], state) do
    {state, ["id name ECHECS-ENGINE", "id author HEKPYTO", "uciok"]}
  end

  defp handle_tokens(["isready"], state), do: {state, ["readyok"]}

  defp handle_tokens(["ucinewgame"], state),
    do: {%{state | game: Echecs.new_game(), history_games: []}, []}

  defp handle_tokens(["position" | tokens], state), do: set_position(state, tokens)

  defp handle_tokens(["go" | tokens], state), do: go(state, tokens)

  defp handle_tokens(["stop"], state), do: stop_search(state)

  defp handle_tokens(["quit"], state) do
    {state, _output} = stop_search(state, emit_bestmove?: false)
    {%{state | quit?: true}, []}
  end

  defp handle_tokens([], state), do: {state, []}

  defp handle_tokens(tokens, state),
    do: {state, ["info string unsupported command #{Enum.join(tokens, " ")}"]}

  defp set_position(state, ["startpos" | rest]) do
    apply_position_moves(state, Echecs.new_game(), moves_after_marker(rest))
  end

  defp set_position(state, ["fen" | rest]) do
    {fen_tokens, move_tokens} = split_fen_and_moves(rest)
    fen = Enum.join(fen_tokens, " ")

    try do
      apply_position_moves(state, Echecs.new_game(fen), move_tokens)
    rescue
      error -> {state, ["info string invalid fen #{Exception.message(error)}"]}
    end
  end

  defp set_position(state, tokens),
    do: {state, ["info string invalid position #{Enum.join(tokens, " ")}"]}

  defp apply_position_moves(state, game, moves) do
    Enum.reduce_while(moves, {:ok, game, []}, fn uci_move, {:ok, current_game, history_games} ->
      case apply_uci_move(current_game, uci_move) do
        {:ok, next_game} -> {:cont, {:ok, next_game, [current_game | history_games]}}
        {:error, _reason} -> {:halt, {:error, uci_move}}
      end
    end)
    |> case do
      {:ok, game, history_games} -> {%{state | game: game, history_games: history_games}, []}
      {:error, move} -> {state, ["info string invalid position move #{move}"]}
    end
  end

  defp go(%__MODULE__{search_task: %Task{}} = state, _tokens),
    do: {state, ["info string search already running"]}

  defp go(%__MODULE__{game: game, history_games: history_games} = state, tokens) do
    fen = Echecs.FEN.to_string(game)

    opts =
      go_opts(tokens, game.turn)
      |> Keyword.put(:history_games, history_games)

    start_search(state, fen, opts)
  end

  defp start_search(%__MODULE__{best_move: best_move} = state, fen, opts) do
    owner = self()
    search_id = make_ref()

    task =
      async_nolink(state.task_supervisor, fn ->
        try do
          send(owner, {:uci_search_started, search_id})

          result =
            best_move.(
              fen,
              opts
              |> Keyword.put(:reporter, reporter_fn(owner, search_id))
            )

          result_to_output(result)
        rescue
          error ->
            result_to_output({:error, error})
        catch
          kind, reason ->
            result_to_output({:error, {kind, reason}})
        end
      end)

    await_search_started(search_id)
    {%{state | search_task: task, search_id: search_id}, []}
  end

  defp await_search_started(search_id) do
    receive do
      {:uci_search_started, ^search_id} -> :ok
    after
      50 -> :ok
    end
  end

  defp stop_search(%__MODULE__{} = state), do: stop_search(state, [])

  defp stop_search(%__MODULE__{} = state, opts) do
    {state, lines} = drain(state)

    case state.search_task do
      nil ->
        {state, lines}

      task ->
        Task.shutdown(task, :brutal_kill)

        final_lines =
          if Keyword.get(opts, :emit_bestmove?, true) and
               not Enum.any?(lines, &String.starts_with?(&1, "bestmove ")) do
            lines ++ ["bestmove 0000"]
          else
            lines
          end

        {%{state | search_task: nil, search_id: nil}, final_lines}
    end
  end

  defp async_nolink(supervisor, fun) do
    case supervisor_pid(supervisor) do
      nil ->
        {:ok, supervisor_pid} = Task.Supervisor.start_link()
        Task.Supervisor.async_nolink(supervisor_pid, fun)

      supervisor_pid ->
        Task.Supervisor.async_nolink(supervisor_pid, fun)
    end
  end

  defp supervisor_pid(supervisor) when is_atom(supervisor), do: Process.whereis(supervisor)
  defp supervisor_pid(supervisor) when is_pid(supervisor), do: supervisor

  defp drain_messages(%__MODULE__{search_id: nil} = state, acc), do: {state, Enum.reverse(acc)}

  defp drain_messages(%__MODULE__{search_task: task, search_id: search_id} = state, acc) do
    receive do
      {:uci_search_started, ^search_id} ->
        drain_messages(state, acc)

      {:uci_search_line, ^search_id, line} ->
        drain_messages(state, [line | acc])

      {ref, lines} when not is_nil(task) and ref == task.ref and is_list(lines) ->
        drop_task_down(task)
        state = %{state | search_task: nil, search_id: nil}
        drain_messages(state, Enum.reverse(lines) ++ acc)

      {ref, _result} when not is_nil(task) and ref == task.ref ->
        drop_task_down(task)
        state = %{state | search_task: nil, search_id: nil}
        drain_messages(state, acc)

      {:DOWN, ref, :process, _pid, _reason} when not is_nil(task) and ref == task.ref ->
        state = %{state | search_task: nil, search_id: nil}
        drain_messages(state, acc)
    after
      0 ->
        {state, Enum.reverse(acc)}
    end
  end

  defp drop_task_down(task) do
    receive do
      {:DOWN, ref, :process, _pid, _reason} when ref == task.ref -> :ok
    after
      0 -> :ok
    end
  end

  defp reporter_fn(owner, search_id),
    do: fn info -> send(owner, {:uci_search_line, search_id, format_info(info)}) end

  defp result_to_output({:ok, move}), do: ["bestmove #{move}"]
  defp result_to_output({:terminal, _status}), do: ["bestmove 0000"]

  defp result_to_output({:error, reason}),
    do: ["info string search error #{inspect(reason)}", "bestmove 0000"]

  defp moves_after_marker(["moves" | moves]), do: moves
  defp moves_after_marker(_tokens), do: []

  defp split_fen_and_moves(tokens) do
    {fen_tokens, rest} = Enum.split_while(tokens, &(&1 != "moves"))

    move_tokens =
      case rest do
        ["moves" | moves] -> moves
        [] -> []
      end

    {Enum.take(fen_tokens, 6), move_tokens}
  end

  defp apply_uci_move(game, <<from::binary-size(2), to::binary-size(2), rest::binary>>) do
    Echecs.make_move(
      game,
      Echecs.Board.to_index(from),
      Echecs.Board.to_index(to),
      parse_promotion(rest)
    )
  rescue
    error -> {:error, error}
  end

  defp apply_uci_move(_game, _move), do: {:error, :invalid_move}

  defp parse_promotion(""), do: nil
  defp parse_promotion("q"), do: :queen
  defp parse_promotion("r"), do: :rook
  defp parse_promotion("b"), do: :bishop
  defp parse_promotion("n"), do: :knight
  defp parse_promotion(_), do: nil

  defp go_opts(tokens, turn) do
    tokens
    |> parse_go_opts([])
    |> derive_movetime(turn)
  rescue
    _error -> []
  end

  defp parse_go_opts([], opts), do: Enum.reverse(opts)

  defp parse_go_opts(["infinite" | rest], opts),
    do: parse_go_opts(rest, [{:infinite, true} | opts])

  defp parse_go_opts(["movetime", value | rest], opts),
    do: parse_go_opts(rest, [{:movetime, String.to_integer(value)} | opts])

  defp parse_go_opts(["depth", value | rest], opts),
    do: parse_go_opts(rest, [{:depth, String.to_integer(value)} | opts])

  defp parse_go_opts(["nodes", value | rest], opts),
    do: parse_go_opts(rest, [{:nodes, String.to_integer(value)} | opts])

  defp parse_go_opts(["wtime", value | rest], opts),
    do: parse_go_opts(rest, [{:wtime, String.to_integer(value)} | opts])

  defp parse_go_opts(["btime", value | rest], opts),
    do: parse_go_opts(rest, [{:btime, String.to_integer(value)} | opts])

  defp parse_go_opts(["winc", value | rest], opts),
    do: parse_go_opts(rest, [{:winc, String.to_integer(value)} | opts])

  defp parse_go_opts(["binc", value | rest], opts),
    do: parse_go_opts(rest, [{:binc, String.to_integer(value)} | opts])

  defp parse_go_opts(["movestogo", value | rest], opts),
    do: parse_go_opts(rest, [{:movestogo, String.to_integer(value)} | opts])

  defp parse_go_opts([_unknown | rest], opts), do: parse_go_opts(rest, opts)

  defp derive_movetime(opts, turn) do
    cond do
      is_integer(opts[:movetime]) ->
        opts

      opts[:infinite] == true ->
        opts

      true ->
        derive_clock_movetime(opts, turn)
    end
  end

  defp derive_clock_movetime(opts, turn) do
    clock = if(turn == :white, do: opts[:wtime], else: opts[:btime])
    increment = if(turn == :white, do: opts[:winc], else: opts[:binc]) || 0

    if is_integer(clock) do
      moves_to_go = max(opts[:movestogo] || 30, 1)
      planned = max(div(clock, moves_to_go) + div(increment * 8, 10), 1)
      safe_budget = if clock > 50, do: min(planned, clock - 50), else: planned
      Keyword.put(opts, :movetime, safe_budget)
    else
      opts
    end
  end

  defp format_info(info) when is_map(info) do
    segments =
      []
      |> maybe_append("depth", info[:depth] || info["depth"])
      |> maybe_append("seldepth", info[:seldepth] || info["seldepth"])
      |> maybe_append("nodes", info[:nodes] || info["nodes"])
      |> maybe_append("time", info[:time] || info["time"])
      |> maybe_append_pv(info[:pv] || info["pv"])

    Enum.join(["info" | segments], " ")
  end

  defp format_info(info), do: "info string #{inspect(info)}"

  defp maybe_append(segments, _label, nil), do: segments
  defp maybe_append(segments, label, value), do: segments ++ [label, to_string(value)]

  defp maybe_append_pv(segments, nil), do: segments
  defp maybe_append_pv(segments, pv) when is_list(pv), do: segments ++ ["pv", Enum.join(pv, " ")]
end
