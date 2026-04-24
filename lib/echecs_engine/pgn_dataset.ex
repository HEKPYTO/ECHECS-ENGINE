defmodule EchecsEngine.PGNDataset do
  @moduledoc """
  Converts PGN game records into supervised JSONL-ready training examples.
  """

  alias Echecs.{FEN, Game, PGN}

  @spec records_from_pgn!(String.t()) :: [map()]
  def records_from_pgn!(pgn) when is_binary(pgn) do
    pgn
    |> split_games()
    |> Enum.flat_map(&records_from_single_game!/1)
  end

  defp records_from_single_game!(pgn) do
    {headers, body} = split_headers_and_body(pgn)
    result = Map.get(headers, "Result", "*")
    san_moves = PGN.parse_moves(body)

    {_game, _history_fens, records} =
      Enum.reduce(san_moves, {Echecs.new_game(), [], []}, fn san, {game, history_fens, records} ->
        fen = FEN.to_string(game)

        move =
          case PGN.move_from_san(game, san) do
            {:ok, move} ->
              move

            {:error, reason} ->
              raise ArgumentError, "invalid SAN #{inspect(san)}: #{inspect(reason)}"
          end

        next_game = Game.make_move(game, move)

        record = %{
          "fen" => fen,
          "move" => move_to_uci(move),
          "result" => result,
          "history_fens" => Enum.take(history_fens, 7)
        }

        {next_game, [fen | history_fens] |> Enum.take(7), [record | records]}
      end)

    Enum.reverse(records)
  end

  @spec write_jsonl!(String.t(), Enumerable.t()) :: :ok
  def write_jsonl!(path, records) do
    path
    |> Path.dirname()
    |> File.mkdir_p!()

    File.open!(path, [:write], fn file ->
      Enum.each(records, fn record ->
        IO.binwrite(file, Jason.encode!(record))
        IO.binwrite(file, "\n")
      end)
    end)
  end

  defp split_headers_and_body(pgn) do
    {header_lines, body_lines} =
      pgn
      |> String.split(~r/\R/, trim: false)
      |> Enum.reduce_while({[], []}, fn line, {headers, body} ->
        cond do
          body != [] ->
            {:cont, {headers, [line | body]}}

          String.trim(line) == "" ->
            {:cont, {headers, [""]}}

          String.starts_with?(line, "[") ->
            {:cont, {[line | headers], body}}

          true ->
            {:cont, {headers, [line | body]}}
        end
      end)

    headers =
      header_lines
      |> Enum.reverse()
      |> Enum.reduce(%{}, fn line, acc ->
        case Regex.run(~r/^\[(\w+)\s+"(.*)"\]$/, String.trim(line)) do
          [_, key, value] -> Map.put(acc, key, value)
          _other -> acc
        end
      end)

    body =
      body_lines
      |> Enum.reverse()
      |> Enum.join("\n")
      |> String.trim()

    {headers, body}
  end

  defp split_games(pgn) do
    {games, current, _seen_body?} =
      pgn
      |> String.split(~r/\R/, trim: false)
      |> Enum.reduce({[], [], false}, fn line, {games, current, seen_body?} ->
        starts_next_game? = String.starts_with?(String.trim(line), "[") and seen_body?

        cond do
          starts_next_game? ->
            {[current_to_game(current) | games], [line], false}

          String.trim(line) == "" ->
            {games, [line | current], seen_body?}

          String.starts_with?(String.trim(line), "[") ->
            {games, [line | current], seen_body?}

          true ->
            {games, [line | current], true}
        end
      end)

    [current_to_game(current) | games]
    |> Enum.reverse()
    |> Enum.reject(&(&1 == ""))
  end

  defp current_to_game(lines) do
    lines
    |> Enum.reverse()
    |> Enum.join("\n")
    |> String.trim()
  end

  defp move_to_uci(%Echecs.Move{} = move) do
    Echecs.Board.to_algebraic(move.from) <>
      Echecs.Board.to_algebraic(move.to) <>
      promotion_suffix(move.promotion)
  end

  defp promotion_suffix(nil), do: ""
  defp promotion_suffix(:queen), do: "q"
  defp promotion_suffix(:rook), do: "r"
  defp promotion_suffix(:bishop), do: "b"
  defp promotion_suffix(:knight), do: "n"
end
