defmodule EchecsEngine.MixProject do
  use Mix.Project

  def project do
    [
      app: :echecs_engine,
      version: "0.1.0",
      elixir: "~> 1.19",
      elixirc_options: [warnings_as_errors: true],
      start_permanent: Mix.env() == :prod,
      dialyzer: [
        plt_file: {:no_warn, "priv/plts/project.plt"},
        plt_add_apps: [:mix]
      ],
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

      # Lint & static analysis (dev only)
      {:credo, "~> 1.7", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: :dev, runtime: false},

      # Security audit (dev only)
      {:mix_audit, "~> 2.0", only: [:dev, :test], runtime: false}
    ]
  end

  defp echecs_dep do
    case System.get_env("ECHECS_PATH") do
      nil -> {:echecs, github: "HEKPYTO/ECHECS"}
      path -> {:echecs, path: path}
    end
  end
end
