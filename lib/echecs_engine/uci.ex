# credo:disable-for-this-file Credo.Check.Refactor.CyclomaticComplexity
# credo:disable-for-this-file Credo.Check.Refactor.Nesting
defmodule EchecsEngine.UCI do
  @moduledoc """
  Universal Chess Interface (UCI) loop for ECHECS-ENGINE.

  Stateful line-protocol adapter between a GUI (or `fastchess`) and
  `EchecsEngine.Search`. The loop is synchronous on stdin/stdout but
  delegates search to `EchecsEngine.SearchSupervisor` so `stop` / `isready`
  / `quit` remain responsive.

  ## Protocol

  Supported commands: `uci`, `isready`, `ucinewgame`, `position` (both
  `startpos` and `fen` with trailing `moves`), `go` (with `depth`/`nodes`/
  `movetime`/`wtime`/`btime`/`winc`/`binc`/`movestogo`/`infinite`), `stop`,
  `quit`. Unknown tokens yield `info string unsupported command`.

  On `go`, the loop spawns an `async_nolink` task that calls
  `EchecsEngine.Search.best_move/2` with `stop_ref` (`:atomics`) and a
  `reporter` that forwards `info` lines by `send/2`. `drain/1` polls the
  mailbox on every input line and every 10 ms so `info` depth lines appear
  outside the input handler.

  Clock-based `go` without an explicit limit is converted to a `movetime`
  allocation: `time / movestogo + 0.75 * increment`, reserving `time / 20`.

  ## State

  `%EchecsEngine.UCI{}` holds `game`, the running `task`/`id`/`stop_ref`,
  `bestmove` fallback (first legal move), `emitted?`, and `quit?`. It is
  exposed so tests can drive `handle_line/2` without I/O.

  ## Example

      iex> state = EchecsEngine.UCI.new()
      iex> {state, out} = EchecsEngine.UCI.handle_line(state, "position startpos moves e2e4")
      iex> {state, out} = EchecsEngine.UCI.handle_line(state, "go depth 1")
      iex> is_struct(state, EchecsEngine.UCI)
      true
  """

  require Echecs.Move

  @mate 32_000

  defstruct game: nil,
            task: nil,
            id: nil,
            stop_ref: nil,
            bestmove: nil,
            emitted?: false,
            quit?: false

  @type t :: %__MODULE__{}

  @doc "Returns a fresh UCI state with the standard start position."
  @spec new() :: t()
  def new, do: %__MODULE__{game: Echecs.new_game()}

  @doc """
  Runs the blocking UCI loop on `stdin`/`stdout`.

  Spawns a dedicated input reader so `stop`/`quit` remain responsive while
  a search task runs under `EchecsEngine.SearchSupervisor`. Returns the
  final state after `quit` or `:eof`.
  """
  @spec run() :: t()
  def run do
    parent = self()
    input_reader = spawn(fn -> read_input(parent) end)
    loop(new(), input_reader)
  end

  @doc """
  Handles a single UCI line against `state`.

  Drains any pending `info`/`bestmove` messages first, then dispatches
  `command/2`. Returns `{new_state, output_lines}` without performing I/O,
  so it is the primary test entry point.
  """
  @spec handle_line(t(), String.t()) :: {t(), [String.t()]}
  def handle_line(state, line) do
    {state, pending} = drain(state)
    {state, output} = command(state, String.split(String.trim(line), ~r/\s+/, trim: true))
    {state, pending ++ output}
  end

  @doc """
  Drains all pending search messages for `state`.

  Collects `{:uci_info, id, info}` → `info ... pv ...` and task completion
  → `bestmove ...` without blocking.
  """
  @spec drain(t()) :: {t(), [String.t()]}
  def drain(%__MODULE__{} = state), do: receive_messages(state, [])

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
        {state, output} = handle_line(state, line)
        print_output(output)

        if state.quit? do
          stop_input_reader(input_reader)
        else
          loop(state, input_reader)
        end

      :uci_eof ->
        {state, output} = handle_line(state, "quit")
        print_output(output)
        stop_input_reader(input_reader)
        state
    after
      10 ->
        {state, output} = drain(state)
        print_output(output)
        loop(state, input_reader)
    end
  end

  defp print_output(output), do: Enum.each(output, &IO.puts/1)

  defp stop_input_reader(input_reader) do
    if Process.alive?(input_reader), do: Process.exit(input_reader, :kill)
  end

  defp command(state, ["uci"]),
    do: {state, ["id name ECHECS-ENGINE", "id author HEKPYTO", "uciok"]}

  defp command(state, ["isready"]), do: {state, ["readyok"]}
  defp command(state, ["ucinewgame"]), do: {%{state | game: Echecs.new_game()}, []}
  defp command(state, ["position" | rest]), do: position(state, rest)

  defp command(%{task: task} = state, ["go" | _]) when not is_nil(task),
    do: {state, ["info string search already running"]}

  defp command(state, ["go" | rest]), do: go(state, rest)
  defp command(state, ["stop"]), do: stop(state)

  defp command(state, ["quit"]) do
    {state, _output} = stop(state)
    {%{state | quit?: true}, []}
  end

  defp command(state, []), do: {state, []}

  defp command(state, tokens),
    do: {state, ["info string unsupported command #{Enum.join(tokens, " ")}"]}

  defp position(state, ["startpos" | rest]),
    do: apply_moves(state, Echecs.new_game(), after_moves(rest))

  defp position(state, ["fen" | rest]) do
    {fen, moves} = split_fen(rest)

    try do
      apply_moves(state, Echecs.new_game(fen), moves)
    rescue
      _ -> {state, ["info string invalid fen"]}
    end
  end

  defp position(state, _), do: {state, ["info string invalid position"]}

  defp apply_moves(state, game, moves) do
    Enum.reduce_while(moves, {:ok, game}, fn text, {:ok, current} ->
      case uci_move(current, text) do
        {:ok, next} -> {:cont, {:ok, next}}
        :error -> {:halt, {:error, text}}
      end
    end)
    |> case do
      {:ok, game} -> {%{state | game: game}, []}
      {:error, text} -> {state, ["info string invalid position move #{text}"]}
    end
  end

  defp go(state, tokens) do
    case go_opts(tokens, state.game.turn) do
      {:error, :invalid_go} ->
        {state, ["info string invalid go"]}

      {:ok, opts} ->
        owner = self()
        id = make_ref()
        stop_ref = :atomics.new(1, [])
        game = state.game

        fallback =
          game |> Echecs.MoveGen.legal_moves_int() |> List.first() |> EchecsEngine.Move.to_uci()

        task =
          Task.Supervisor.async_nolink(EchecsEngine.SearchSupervisor, fn ->
            case EchecsEngine.Search.best_move(
                   game,
                   opts ++
                     [
                       stop_ref: stop_ref,
                       reporter: fn info -> send(owner, {:uci_info, id, info}) end
                     ]
                 ) do
              {:ok, move, info} ->
                {:ok, Map.put(info, :bestmove, EchecsEngine.Move.to_uci(move))}

              other ->
                other
            end
          end)

        {%{state | task: task, id: id, stop_ref: stop_ref, bestmove: fallback, emitted?: false},
         []}
    end
  end

  defp go_opts(tokens, turn) do
    with {:ok, controls} <- parse_go(tokens, %{}), do: search_limit(controls, turn)
  end

  defp parse_go([], controls), do: {:ok, controls}
  defp parse_go(["infinite"], controls), do: put_control(controls, :infinite, true)

  defp parse_go([name, value | rest], controls)
       when name in ["depth", "nodes", "movetime", "movestogo"] do
    with {:ok, number} <- parse_int(value),
         {:ok, controls} <- put_control(controls, String.to_existing_atom(name), number) do
      parse_go(rest, controls)
    end
  end

  defp parse_go([name, value | rest], controls)
       when name in ["wtime", "btime", "winc", "binc"] do
    with {:ok, number} <- parse_nonnegative_int(value),
         {:ok, controls} <- put_control(controls, String.to_existing_atom(name), number) do
      parse_go(rest, controls)
    end
  end

  defp parse_go(_, _), do: {:error, :invalid_go}

  defp put_control(controls, key, value) do
    if Map.has_key?(controls, key),
      do: {:error, :invalid_go},
      else: {:ok, Map.put(controls, key, value)}
  end

  defp search_limit(controls, turn) do
    explicit = Enum.filter([:depth, :nodes, :movetime, :infinite], &Map.has_key?(controls, &1))

    case explicit do
      [:infinite] when map_size(controls) == 1 ->
        {:ok, [nodes: 9_223_372_036_854_775_807]}

      [key] when key != :infinite ->
        {:ok, [{key, Map.fetch!(controls, key)}]}

      [] when map_size(controls) == 0 ->
        {:ok, []}

      [] ->
        clock_limit(controls, turn)

      _ ->
        {:error, :invalid_go}
    end
  end

  defp clock_limit(controls, turn) do
    {time_key, increment_key} = if turn == :white, do: {:wtime, :winc}, else: {:btime, :binc}

    case Map.fetch(controls, time_key) do
      {:ok, time} ->
        moves = Map.get(controls, :movestogo, 30)
        increment = Map.get(controls, increment_key, 0)
        allocation = div(time, moves) + div(increment * 3, 4)
        reserve = max(div(time, 20), 1)
        {:ok, [movetime: min(max(allocation, 1), max(time - reserve, 1))]}

      :error ->
        {:error, :invalid_go}
    end
  end

  defp parse_int(value) do
    case Integer.parse(value) do
      {number, ""} when number > 0 -> {:ok, number}
      _ -> {:error, :invalid_go}
    end
  end

  defp parse_nonnegative_int(value) do
    case Integer.parse(value) do
      {number, ""} when number >= 0 -> {:ok, number}
      _ -> {:error, :invalid_go}
    end
  end

  defp stop(%{task: nil} = state), do: {state, []}

  defp stop(state) do
    :atomics.put(state.stop_ref, 1, 1)
    {state, lines} = await_task(state, [], System.monotonic_time(:millisecond) + 100)

    state =
      case state.task do
        nil ->
          state

        task ->
          _ = Task.shutdown(task, :brutal_kill)
          clear_task(state)
      end

    if state.emitted? do
      {state, lines}
    else
      {%{state | emitted?: true}, lines ++ ["bestmove #{state.bestmove}"]}
    end
  end

  defp receive_messages(%{task: nil} = state, lines), do: {state, Enum.reverse(lines)}

  defp receive_messages(state, lines) do
    receive do
      {:uci_info, id, info} when id == state.id ->
        bestmove = List.first(info.pv) || state.bestmove
        receive_messages(%{state | bestmove: bestmove}, [format_info(info) | lines])

      {ref, result} when ref == state.task.ref ->
        {state, output} = completed(state, result)
        state = clear_task(state)
        receive_messages(state, Enum.reverse(output) ++ lines)

      {:DOWN, ref, :process, _pid, _} when ref == state.task.ref ->
        {state, output} = completed(state, {:error, :task_down})
        state = clear_task(state)
        receive_messages(state, Enum.reverse(output) ++ lines)
    after
      0 -> {state, Enum.reverse(lines)}
    end
  end

  defp await_task(%{task: nil} = state, lines, _deadline), do: {state, lines}

  defp await_task(state, lines, deadline) do
    remaining = deadline - System.monotonic_time(:millisecond)

    if remaining <= 0 do
      {state, lines}
    else
      receive do
        {:uci_info, id, info} when id == state.id ->
          bestmove = List.first(info.pv) || state.bestmove
          await_task(%{state | bestmove: bestmove}, lines ++ [format_info(info)], deadline)

        {ref, result} when ref == state.task.ref ->
          {state, output} = completed(state, result)
          state = clear_task(state)
          {state, lines ++ output}

        {:DOWN, ref, :process, _pid, _} when ref == state.task.ref ->
          {state, output} = completed(state, {:error, :task_down})
          state = clear_task(state)
          {state, lines ++ output}
      after
        remaining -> {state, lines}
      end
    end
  end

  defp completed(state, {:ok, %{bestmove: move}}) do
    output = if(state.emitted?, do: [], else: ["bestmove #{move}"])
    {%{state | bestmove: move, emitted?: true}, output}
  end

  defp completed(state, {:terminal, _}) do
    output = if(state.emitted?, do: [], else: ["bestmove 0000"])
    {%{state | emitted?: true}, output}
  end

  defp completed(state, {:error, reason}) do
    output =
      if state.emitted? do
        []
      else
        ["info string search error #{inspect(reason)}", "bestmove #{state.bestmove || "0000"}"]
      end

    {%{state | emitted?: true}, output}
  end

  defp completed(state, result) do
    completed(state, {:error, result})
  end

  defp clear_task(state) do
    Process.demonitor(state.task.ref, [:flush])
    %{state | task: nil, id: nil, stop_ref: nil}
  end

  defp format_info(info),
    do:
      "info depth #{info.depth} seldepth #{info.seldepth} score #{score(info.score)} nodes #{info.nodes} time #{info.time_ms} nps #{nps(info.nodes, info.time_ms)} pv #{Enum.join(info.pv, " ")}"

  defp score(score) when score > 31_000, do: "mate #{div(@mate - score + 1, 2)}"
  defp score(score) when score < -31_000, do: "mate #{-div(@mate + score, 2)}"
  defp score(score), do: "cp #{score}"
  defp nps(_, 0), do: 0
  defp nps(nodes, ms), do: div(nodes * 1_000, ms)

  defp after_moves(["moves" | moves]), do: moves
  defp after_moves(_), do: []

  defp split_fen(tokens) do
    {fen, rest} = Enum.split_while(tokens, &(&1 != "moves"))
    {Enum.join(fen, " "), after_moves(rest)}
  end

  defp uci_move(game, <<from::binary-size(2), to::binary-size(2), promotion::binary>>) do
    promotion =
      case promotion do
        "" -> nil
        "q" -> :queen
        "r" -> :rook
        "b" -> :bishop
        "n" -> :knight
        _ -> :invalid
      end

    move =
      Enum.find(Echecs.MoveGen.legal_moves_int(game), fn packed ->
        Echecs.Board.to_algebraic(Echecs.Move.unpack_from(packed)) == from and
          Echecs.Board.to_algebraic(Echecs.Move.unpack_to(packed)) == to and
          Echecs.Move.unpack_promotion(packed) == promotion
      end)

    if move, do: {:ok, Echecs.Game.make_move_int(game, move)}, else: :error
  end

  defp uci_move(_, _), do: :error
end
