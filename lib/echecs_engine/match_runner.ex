defmodule EchecsEngine.MatchRunner do
  @moduledoc """
  Builds and runs external engine-vs-engine match commands.

  ECHECS-ENGINE keeps match execution outside the VM and delegates to established
  UCI match runners such as `cutechess-cli` and `fastchess`.
  """

  @type engine :: %{required(:name) => String.t(), required(:command) => String.t()}

  @spec command(keyword()) :: {String.t(), [String.t()]}
  def command(opts) do
    if Keyword.get(opts, :docker, false) do
      docker_command(opts)
    else
      runner = Keyword.get(opts, :runner, "cutechess-cli")

      case runner do
        "fastchess" -> fastchess_command(opts)
        _other -> cutechess_command(runner, opts)
      end
    end
  end

  @spec run(keyword()) ::
          {:ok, %{status: non_neg_integer(), output: String.t()}} | {:error, term()}
  def run(opts) do
    {exe, args} = command(opts)

    case System.find_executable(exe) do
      nil ->
        {:error, {:runner_not_found, exe}}

      _path ->
        {output, status} = System.cmd(exe, args, stderr_to_stdout: true)
        {:ok, %{status: status, output: output}}
    end
  end

  @spec shell_command(keyword()) :: String.t()
  def shell_command(opts) do
    {exe, args} = command(opts)

    ([exe] ++ args)
    |> Enum.map(&shell_escape/1)
    |> Enum.join(" ")
  end

  defp cutechess_command(runner, opts) do
    engine_a = engine!(opts, :engine_a)
    engine_b = engine!(opts, :engine_b)
    games = Keyword.get(opts, :games, 100)
    concurrency = Keyword.get(opts, :concurrency, System.schedulers_online())
    tc = Keyword.get(opts, :tc, "10+0.1")

    args =
      [
        "-engine",
        "name=#{engine_a.name}",
        "cmd=#{engine_a.command}",
        "-engine",
        "name=#{engine_b.name}",
        "cmd=#{engine_b.command}",
        "-each",
        "proto=uci",
        "tc=#{tc}",
        "-games",
        to_string(games),
        "-concurrency",
        to_string(concurrency),
        "-repeat"
      ]
      |> maybe_openings(opts)
      |> maybe_sprt(opts)

    {runner, args}
  end

  defp fastchess_command(opts) do
    engine_a = engine!(opts, :engine_a)
    engine_b = engine!(opts, :engine_b)
    games = Keyword.get(opts, :games, 100)
    concurrency = Keyword.get(opts, :concurrency, System.schedulers_online())
    tc = Keyword.get(opts, :tc, "10+0.1")
    rounds = max(div(games, 2), 1)

    args =
      [
        "-engine",
        "name=#{engine_a.name}",
        "cmd=#{engine_a.command}",
        "-engine",
        "name=#{engine_b.name}",
        "cmd=#{engine_b.command}",
        "-each",
        "proto=uci",
        "tc=#{tc}",
        "-rounds",
        to_string(rounds),
        "-concurrency",
        to_string(concurrency),
        "-repeat"
      ]
      |> maybe_openings(opts)
      |> maybe_sprt(opts)

    {"fastchess", args}
  end

  defp docker_command(opts) do
    runner = Keyword.get(opts, :runner, "fastchess")

    if runner != "fastchess" do
      raise ArgumentError, "docker match runner supports only fastchess"
    end

    service = Keyword.get(opts, :docker_service, "engine-match")
    {"fastchess", fastchess_args} = fastchess_command(opts)

    {"docker", ["compose", "run", "--rm", "--no-deps", service, "fastchess" | fastchess_args]}
  end

  defp engine!(opts, key) do
    case Keyword.fetch!(opts, key) do
      %{name: name, command: command} -> %{name: name, command: command}
      %{"name" => name, "command" => command} -> %{name: name, command: command}
    end
  end

  defp maybe_openings(args, opts) do
    case Keyword.get(opts, :openings) do
      nil -> args
      path -> args ++ ["-openings", "file=#{path}"]
    end
  end

  defp maybe_sprt(args, opts) do
    case Keyword.get(opts, :sprt) do
      nil ->
        args

      %{elo0: elo0, elo1: elo1, alpha: alpha, beta: beta} ->
        args ++
          [
            "-sprt",
            "elo0=#{elo0}",
            "elo1=#{elo1}",
            "alpha=#{alpha}",
            "beta=#{beta}"
          ]
    end
  end

  defp shell_escape(value) do
    if String.match?(value, ~r/^[A-Za-z0-9_.,:+=@%\/-]+$/) do
      value
    else
      "'" <> String.replace(value, "'", "'\"'\"'") <> "'"
    end
  end
end
