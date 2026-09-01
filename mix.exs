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
      docs: docs(),
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

      # Docs
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},

      # Lint & static analysis (dev only)
      {:credo, "~> 1.7", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: :dev, runtime: false},

      # Security audit (dev only)
      {:mix_audit, "~> 2.0", only: [:dev, :test], runtime: false}
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md"],
      groups_for_modules: [
        Engine: [EchecsEngine, EchecsEngine.Search, EchecsEngine.Eval, EchecsEngine.Move],
        Protocol: [EchecsEngine.UCI],
        Training: [EchecsEngine.Trainer, EchecsEngine.Bench],
        Internals: [EchecsEngine.Application],
        Tasks: [
          Mix.Tasks.Engine.Bench,
          Mix.Tasks.Engine.Best,
          Mix.Tasks.Engine.Seed,
          Mix.Tasks.Engine.TrainEvaluator,
          Mix.Tasks.Engine.Uci
        ]
      ],
      groups_for_extras: [Guides: ~r/README\.md/],
      source_url: "https://github.com/HEKPYTO/ECHECS-ENGINE"
    ]
  end

  defp echecs_dep do
    case System.get_env("ECHECS_PATH") do
      nil -> {:echecs, github: "HEKPYTO/ECHECS"}
      path -> {:echecs, path: path}
    end
  end
end
