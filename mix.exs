defmodule EchecsZero.MixProject do
  use Mix.Project

  def project do
    [
      app: :echecs_zero,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {EchecsZero.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:echecs, github: "HEKPYTO/ECHECS"},
      {:nx, "~> 0.7"},
      {:axon, "~> 0.6"},
      {:exla, "~> 0.7"}
    ]
  end
end
