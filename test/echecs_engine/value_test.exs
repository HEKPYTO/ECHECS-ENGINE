defmodule EchecsEngine.ValueTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.Value

  @start_fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  test "uses explicit calibrated WDL labels when present" do
    game = Echecs.new_game(@start_fen)

    wdl = Value.target_from_record(%{"wdl" => [0.2, 0.7, 0.1], "result" => "1-0"}, game)

    assert_in_delta Nx.to_number(wdl[0]), 0.2, 1.0e-6
    assert_in_delta Nx.to_number(wdl[1]), 0.7, 1.0e-6
    assert_in_delta Nx.to_number(wdl[2]), 0.1, 1.0e-6
    assert_in_delta Nx.sum(wdl) |> Nx.to_number(), 1.0, 1.0e-6
  end

  test "centipawn targets keep a non-zero draw channel" do
    game = Echecs.new_game(@start_fen)

    wdl = Value.target_from_record(%{"eval_cp" => 80}, game)

    assert Nx.to_number(wdl[0]) > Nx.to_number(wdl[2])
    assert Nx.to_number(wdl[1]) > 0.0
    assert_in_delta Nx.sum(wdl) |> Nx.to_number(), 1.0, 1.0e-6
  end

  test "centipawn targets preserve calibrated draw probability instead of q-only expansion" do
    game = Echecs.new_game(@start_fen)

    wdl = Value.target_from_record(%{"eval_cp" => 100}, game)
    q_only = Value.q_to_wdl(Value.wdl_to_q(wdl))

    refute_in_delta Nx.to_number(wdl[1]), Nx.to_number(q_only[1]), 1.0e-6
  end

  test "uses explicit eval_wdl labels when present" do
    game = Echecs.new_game(@start_fen)

    wdl =
      Value.target_from_record(%{"eval_wdl" => %{"win" => 12, "draw" => 70, "loss" => 18}}, game)

    assert_in_delta Nx.to_number(wdl[0]), 0.12, 1.0e-6
    assert_in_delta Nx.to_number(wdl[1]), 0.70, 1.0e-6
    assert_in_delta Nx.to_number(wdl[2]), 0.18, 1.0e-6
  end

  test "centipawn fallback calibration depends on game progress" do
    early = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w - - 0 8")
    late = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w - - 0 40")

    early_wdl = Value.target_from_record(%{"eval_cp" => 100}, early)
    late_wdl = Value.target_from_record(%{"eval_cp" => 100}, late)

    refute early_wdl |> Nx.to_flat_list() == late_wdl |> Nx.to_flat_list()
  end
end
