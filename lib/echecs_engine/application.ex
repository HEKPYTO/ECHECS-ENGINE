defmodule EchecsEngine.Application do
  @moduledoc false
  # Internal OTP application callback.
  #
  # Starts the single `Task.Supervisor` that hosts UCI `go` searches.
  # Hidden from ExDoc (`@moduledoc false`) because it is not part of the
  # public API; documented here for maintainability.

  use Application

  @impl true
  def start(_type, _args) do
    children = [{Task.Supervisor, name: EchecsEngine.SearchSupervisor}]

    opts = [strategy: :one_for_one, name: EchecsEngine.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
