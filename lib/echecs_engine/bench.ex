# credo:disable-for-this-file Credo.Check.Refactor.CyclomaticComplexity
defmodule EchecsEngine.Bench do
  @moduledoc """
  Deterministic correctness and measurement helpers.

  Provides three locally reproducible measurements plus one externally
  delegated strength gate:

  * `signature/1` - runs the fixed two-position `@corpus` through
    `EchecsEngine.analyze/2` and returns `nodes`, `nps`, `gc`, `memory`,
    `tt` stats and PVs. `mix engine.bench` calls `smoke/0` (`depth: 2`)
    as a fast sanity check; any `opts` (e.g. `depth: 4`) are forwarded.
  * `run_jsonl!/2` - streams a `fen` / optional `bestmove` JSONL file and
    reports `solved/total` plus per-position info. Used for fixture-based
    regression.
  * `parse_fastchess_report!/1` - parses the *final* fastchess
    `Ptnml(0-2)` + `LLR` normalized-Elo SPRT block. No live score
    scraping; only the completed report is trusted.
  * `run_fastchess!/2` - shells out to an externally installed `fastchess`
    binary with `base_cmd`/`candidate_cmd` and an opening book, then feeds
    stdout to `parse_fastchess_report!/1`.

  `signature/1` and `run_jsonl!/2` are the correctness baselines shipped
  with the repo. Strength claims require `run_fastchess!/2` and an opening
  book; the module never claims Stockfish parity from `signature` alone.
  """

  @corpus [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3"
  ]

  @fastchess_report ~r/Ptnml\(0-2\):\s*\[\s*(?<b0>\d+)\s*,\s*(?<b1>\d+)\s*,\s*(?<b2>\d+)\s*,\s*(?<b3>\d+)\s*,\s*(?<b4>\d+)\s*\]\s*(?:(?!Ptnml\(0-2\)).)*?LLR:\s*(?<llr>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\(\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)%\s*\)\s*\(\s*(?<lower>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*(?<upper>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*\[\s*(?<elo0>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*(?<elo1>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]/s

  @doc """
  Runs the fixed corpus through the engine and returns a measurement map.

  Includes OTP version, corpus checksum, nodes/nps, GC delta, TT stats,
  and per-position scores/PVs. `opts` are forwarded to `EchecsEngine.analyze/2`.
  """
  @spec signature(keyword()) :: map()
  def signature(opts \\ []) do
    memory_before = :erlang.memory(:total)
    gc_before = :erlang.statistics(:garbage_collection)
    {elapsed, results} = :timer.tc(fn -> Enum.map(@corpus, &EchecsEngine.analyze(&1, opts)) end)
    memory_after = :erlang.memory(:total)
    gc_after = :erlang.statistics(:garbage_collection)
    infos = for {:ok, info} <- results, do: info
    nodes = Enum.sum(Enum.map(infos, & &1.nodes))
    tt_hits = Enum.sum(Enum.map(infos, & &1.tt_hits))
    tt_slots = Enum.sum(Enum.map(infos, & &1.tt_slots))
    tt_entries = Enum.sum(Enum.map(infos, & &1.tt_entries))

    %{
      engine_version: "pure-elixir-1",
      otp: System.otp_release(),
      corpus_checksum:
        :crypto.hash(:sha256, Enum.join(@corpus, "\n")) |> Base.encode16(case: :lower),
      positions: length(infos),
      nodes: nodes,
      nps: if(elapsed == 0, do: 0, else: div(nodes * 1_000_000, elapsed)),
      memory_bytes_delta: memory_after - memory_before,
      gc: gc_delta(gc_before, gc_after),
      tt_hits: tt_hits,
      tt_hit_rate: tt_hits / max(nodes, 1),
      tt_slots: tt_slots,
      tt_entries: tt_entries,
      score: Enum.map(infos, & &1.score),
      pv: Enum.map(infos, & &1.pv)
    }
  end

  @doc "Fast smoke signature at `depth: 2`."
  @spec smoke() :: map()
  def smoke, do: signature(depth: 2)

  @doc """
  Streams a JSONL file of `%{"fen" => fen, "bestmove" => uci | nil}` rows.

  Returns `%{total: total, solved: solved, positions: positions, accuracy: accuracy}` where `positions` holds each engine
  info enriched with `:correct` and `accuracy` is `solved / total` (`0.0` when `total` is `0`).
  Raises `ArgumentError` on malformed rows or engine errors. `opts` are forwarded to `EchecsEngine.analyze/2`.
  """
  @spec run_jsonl!(Path.t(), keyword()) :: %{
          total: non_neg_integer(),
          solved: non_neg_integer(),
          positions: [map()],
          accuracy: float()
        }
  def run_jsonl!(path, opts \\ []) do
    File.stream!(path)
    |> Stream.with_index(1)
    |> Enum.reduce(%{total: 0, solved: 0, positions: []}, fn {line, number}, acc ->
      row = decode_row!(line, number)
      fen = Map.fetch!(row, "fen")

      case EchecsEngine.analyze(fen, opts) do
        {:ok, info} ->
          correct = is_nil(row["bestmove"]) or row["bestmove"] == info.bestmove

          %{
            acc
            | total: acc.total + 1,
              solved: acc.solved + if(correct, do: 1, else: 0),
              positions: [Map.put(info, :correct, correct) | acc.positions]
          }

        other ->
          raise ArgumentError, "line #{number}: #{inspect(other)}"
      end
    end)
    |> then(fn result ->
      Map.put(
        result,
        :accuracy,
        if(result.total == 0, do: 0.0, else: result.solved / result.total)
      )
    end)
  end

  @doc "Parses the final official fastchess normalized-Elo SPRT report from stdout."
  @spec parse_fastchess_report!(binary()) :: map()
  def parse_fastchess_report!(stdout) when is_binary(stdout) do
    case Regex.scan(@fastchess_report, stdout) |> List.last() do
      nil ->
        raise ArgumentError, "fastchess output did not contain a complete penta SPRT report"

      _ ->
        captures =
          Regex.named_captures(
            @fastchess_report,
            Regex.scan(@fastchess_report, stdout) |> List.last() |> hd()
          )

        buckets = Enum.map(0..4, &(captures["b#{&1}"] |> parse_integer!()))
        llr = parse_number!(captures["llr"])
        lower = parse_number!(captures["lower"])
        upper = parse_number!(captures["upper"])

        %{
          buckets: buckets,
          llr: llr,
          lower_bound: lower,
          upper_bound: upper,
          elo0: parse_number!(captures["elo0"]),
          elo1: parse_number!(captures["elo1"]),
          decision: decision(llr, lower, upper)
        }
    end
  end

  def parse_fastchess_report!(_), do: raise(ArgumentError, "fastchess output must be text")

  @doc "Runs an externally installed fastchess paired match and parses its official report."
  @spec run_fastchess!(keyword(), keyword()) :: map()
  def run_fastchess!(match_opts, runtime_opts \\ [])
      when is_list(match_opts) and is_list(runtime_opts) do
    opts =
      match_opts
      |> Keyword.merge(runtime_opts)
      |> Keyword.put_new(:rounds, 100)
      |> Keyword.put_new(:tc, "10+0.1")
      |> Keyword.put_new(:concurrency, 1)
      |> Keyword.put_new(:elo0, 0.0)
      |> Keyword.put_new(:elo1, 5.0)

    validate_match_opts!(opts)
    executable = Keyword.get(opts, :executable, "fastchess")
    runner = Keyword.get(opts, :runner, &System.cmd/3)
    cmd_opts = [stderr_to_stdout: true] ++ if(opts[:dir], do: [cd: opts[:dir]], else: [])
    {stdout, status} = runner.(executable, fastchess_argv(opts), cmd_opts)

    if status == 0 do
      parse_fastchess_report!(stdout)
    else
      raise ArgumentError, "fastchess exited #{status}: #{String.trim(stdout)}"
    end
  end

  defp fastchess_argv(opts) do
    engine = fn name, command, args ->
      ["-engine", "name=#{name}", "cmd=#{command}", "args=#{Enum.join(args, " ")}"]
    end

    engine.(:base, opts[:base_cmd], opts[:base_args]) ++
      engine.(:candidate, opts[:candidate_cmd], opts[:candidate_args]) ++
      [
        "-each",
        "tc=#{opts[:tc]}",
        "-rounds",
        Integer.to_string(opts[:rounds]),
        "-repeat",
        "-concurrency",
        Integer.to_string(opts[:concurrency]),
        "-openings",
        "file=#{opts[:book]}",
        "format=#{book_format(opts[:book])}",
        "order=random",
        "-report",
        "penta=true",
        "-sprt",
        "elo0=#{opts[:elo0]}",
        "elo1=#{opts[:elo1]}",
        "alpha=0.05",
        "beta=0.05",
        "model=normalized"
      ]
  end

  defp validate_match_opts!(opts) do
    required = [:base_cmd, :candidate_cmd, :book]

    Enum.each(required, fn key ->
      unless is_binary(opts[key]) and opts[key] != "", do: raise(ArgumentError, "missing #{key}")
    end)

    unless Enum.all?([:base_args, :candidate_args], &string_args?(opts[&1])),
      do: raise(ArgumentError, "engine arguments must be string lists")

    rounds = Keyword.get(opts, :rounds, 100)
    concurrency = Keyword.get(opts, :concurrency, 1)
    tc = Keyword.get(opts, :tc, "10+0.1")
    elo0 = Keyword.get(opts, :elo0, 0.0)
    elo1 = Keyword.get(opts, :elo1, 5.0)

    unless is_integer(rounds) and rounds > 0 and is_integer(concurrency) and concurrency > 0 and
             is_binary(tc) and tc != "" and is_number(elo0) and is_number(elo1) and elo1 > elo0,
           do: raise(ArgumentError, "invalid fastchess match options")
  end

  defp string_args?(args) when is_list(args), do: Enum.all?(args, fn arg -> is_binary(arg) end)
  defp string_args?(_), do: false

  defp book_format(book) do
    case book |> Path.extname() |> String.downcase() do
      ".epd" -> "epd"
      _ -> "pgn"
    end
  end

  defp decision(llr, _lower, upper) when llr >= upper, do: :accept
  defp decision(llr, lower, _upper) when llr <= lower, do: :reject
  defp decision(_, _, _), do: :continue

  defp parse_integer!(value) do
    case Integer.parse(value) do
      {integer, ""} when integer >= 0 -> integer
      _ -> raise ArgumentError, "fastchess penta buckets must be non-negative integers"
    end
  end

  defp parse_number!(value) do
    case Float.parse(value) do
      {number, ""} -> number
      _ -> raise ArgumentError, "fastchess report contained an invalid number"
    end
  end

  defp gc_delta({collections, words, _}, {after_collections, after_words, _}) do
    %{
      collections_delta: after_collections - collections,
      words_reclaimed_delta: after_words - words
    }
  end

  defp decode_row!(line, number) do
    case Jason.decode(String.trim(line)) do
      {:ok, %{"fen" => _} = row} -> row
      {:ok, _} -> raise ArgumentError, "line #{number}: expected JSON object with fen"
      {:error, _} -> raise ArgumentError, "line #{number}: invalid JSON"
    end
  end
end
