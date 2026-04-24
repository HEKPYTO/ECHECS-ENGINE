defmodule EchecsEngine.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {Task.Supervisor, name: EchecsEngine.SearchTaskSupervisor},
      EchecsEngine.Serving
    ]

    opts = [strategy: :one_for_one, name: EchecsEngine.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
