defmodule EchecsEngine.MixProject do
  use Mix.Project

  def project do
    [
      app: :echecs_engine,
      version: "0.1.0",
      elixir: "~> 1.19",
      elixirc_options: [warnings_as_errors: true],
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {EchecsEngine.Application, []}
    ]
  end

  defp deps do
    [
      echecs_dep(),
      {:jason, "~> 1.4"},
      {:nx, "~> 0.7"},
      {:axon, "~> 0.6"},
      {:exla, "~> 0.7"}
    ]
  end

  defp echecs_dep do
    case System.get_env("ECHECS_PATH") do
      nil -> {:echecs, github: "HEKPYTO/ECHECS"}
      path -> {:echecs, path: path}
    end
  end
end
