defmodule EchecsEngine.NNUE.AccumulatorTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.NNUE.Accumulator

  test "refresh builds the sparse evaluator accumulator shape" do
    game = Echecs.new_game()
    acc = Accumulator.refresh(game, feature_table: feature_table_for(game))

    assert Nx.shape(acc) == {3072}
    assert Nx.type(acc) == {:f, 32}
    assert Nx.sum(acc) |> Nx.to_number() > 0.0
  end

  test "incremental update matches refresh after a legal move" do
    game = Echecs.new_game()
    move = Enum.find(Echecs.legal_moves(game), &(&1.from == 52 and &1.to == 36))
    table = feature_table_for(game)
    before = Accumulator.refresh(game, feature_table: table)

    {:ok, next_game} = Echecs.make_move(game, move.from, move.to, move.promotion)

    updated = Accumulator.update_after_move(before, game, move, feature_table: table)
    refreshed = Accumulator.refresh(next_game, feature_table: table)

    assert Nx.equal(updated, refreshed) |> Nx.all() |> Nx.to_number() == 1
  end

  test "incremental update matches refresh after captures promotions and king moves" do
    scenarios = [
      {"8/8/8/3p4/4P3/8/8/4K2k w - - 0 1", fn move -> move.from == 36 and move.to == 27 end},
      {"8/P7/8/8/8/8/8/4K2k w - - 0 1", fn move -> move.promotion == :queen end},
      {"8/8/8/8/8/8/4P3/4K2k w - - 0 1", fn move -> move.from == 60 and move.to == 59 end}
    ]

    Enum.each(scenarios, fn {fen, selector} ->
      game = Echecs.new_game(fen)
      move = Enum.find(Echecs.legal_moves(game), selector)
      {:ok, next_game} = Echecs.make_move(game, move.from, move.to, move.promotion)
      table = feature_table_for_positions([game, next_game])
      before = Accumulator.refresh(game, feature_table: table)

      updated = Accumulator.update_after_move(before, game, move, feature_table: table)
      refreshed = Accumulator.refresh(next_game, feature_table: table)

      assert Nx.equal(updated, refreshed) |> Nx.all() |> Nx.to_number() == 1
    end)
  end

  test "refresh without a feature table raises instead of hashing placeholders" do
    assert_raise ArgumentError, ~r/feature_table is required/, fn ->
      Accumulator.refresh(Echecs.new_game())
    end
  end

  test "learned feature tables preserve exact feature identity without hashing" do
    game = Echecs.new_game()
    feature_indices = EchecsEngine.NNUE.Features.active_indices(game).white
    first_feature = List.first(feature_indices)

    table = %{
      first_feature => Nx.broadcast(0.0, {3072}) |> Nx.put_slice([17], Nx.tensor([3.5]))
    }

    acc = Accumulator.refresh(game, feature_table: table)

    assert Nx.to_number(acc[17]) == 3.5
    assert Nx.sum(acc) |> Nx.to_number() == 3.5
  end

  test "compact learned feature rows are accumulated without full tensors" do
    game = Echecs.new_game()
    feature_indices = EchecsEngine.NNUE.Features.active_indices(game).white
    [first_feature, second_feature | _rest] = feature_indices

    table = %{
      first_feature => %{indices: [11, 19], values: [1.25, -0.25]},
      second_feature => %{indices: [11, 31], values: [0.75, 0.5]}
    }

    acc = Accumulator.refresh(game, feature_table: table)

    assert Nx.to_number(acc[11]) == 2.0
    assert Nx.to_number(acc[19]) == -0.25
    assert Nx.to_number(acc[31]) == 0.5
  end

  test "public engine accumulator uses NNUE refresh" do
    game = Echecs.new_game()
    table = feature_table_for(game)

    assert Nx.equal(
             EchecsEngine.Accumulator.from_game(game, feature_table: table),
             Accumulator.refresh(game, feature_table: table)
           )
           |> Nx.all()
           |> Nx.to_number() == 1
  end

  defp feature_table_for(game) do
    feature_table_for_positions([game])
  end

  defp feature_table_for_positions(games) do
    games
    |> Enum.flat_map(fn game ->
      game
      |> EchecsEngine.NNUE.Features.active_indices()
      |> Map.values()
      |> List.flatten()
    end)
    |> Enum.uniq()
    |> Enum.map(fn feature_idx ->
      bucket = rem(feature_idx, 3072)
      contribution = Nx.broadcast(0.0, {3072}) |> Nx.put_slice([bucket], Nx.tensor([1.0]))
      {feature_idx, contribution}
    end)
    |> Map.new()
  end
end
