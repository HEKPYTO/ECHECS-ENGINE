checkpoint_path = "models/echecs_engine_latest.axon"
state = :erlang.binary_to_term(File.read!(checkpoint_path))
IO.inspect(Nx.backend(state["patch_embedding"]["kernel"]))
