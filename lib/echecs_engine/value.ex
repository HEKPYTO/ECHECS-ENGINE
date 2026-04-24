defmodule EchecsEngine.Value do
  @moduledoc """
  Converts between chess evaluation targets and WDL/Q values.
  """

  @spec wdl_to_q(Nx.Tensor.t() | [number()]) :: float()
  def wdl_to_q(%Nx.Tensor{} = wdl) do
    win = wdl[0] |> Nx.to_number()
    loss = wdl[2] |> Nx.to_number()
    win - loss
  end

  def wdl_to_q([win, _draw, loss]), do: win - loss

  @spec output_to_q(map()) :: float()
  def output_to_q(%{wdl: wdl}), do: wdl_to_q(wdl)

  def output_to_q(%{value: value}) do
    value
    |> Nx.squeeze()
    |> Nx.to_number()
  end

  @spec target_from_record(map(), Echecs.Game.t()) :: Nx.Tensor.t()
  def target_from_record(record, game) do
    cond do
      Map.has_key?(record, "wdl") and is_list(record["wdl"]) ->
        record["wdl"] |> normalize_wdl() |> Nx.tensor(type: :f32)

      Map.has_key?(record, "eval_wdl") ->
        record["eval_wdl"] |> normalize_eval_wdl() |> Nx.tensor(type: :f32)

      Map.has_key?(record, "eval_cp") and not is_nil(record["eval_cp"]) ->
        cp_to_wdl(record["eval_cp"], game)

      true ->
        record
        |> record_to_q(game)
        |> q_to_wdl()
    end
  end

  defp record_to_q(record, game) do
    cond do
      Map.has_key?(record, "mate") and not is_nil(record["mate"]) ->
        mate_to_q(record["mate"])

      true ->
        result_to_q(record["result"], game.turn)
    end
  end

  @spec q_to_wdl(float()) :: Nx.Tensor.t()
  def q_to_wdl(q) do
    q = max(-1.0, min(1.0, q))
    win = max(q, 0.0)
    loss = max(-q, 0.0)
    draw = 1.0 - abs(q)

    Nx.tensor([win, draw, loss], type: :f32)
  end

  defp normalize_wdl([win, draw, loss]) do
    total = win + draw + loss

    if total > 0.0 do
      [win / total, draw / total, loss / total]
    else
      [0.0, 1.0, 0.0]
    end
  end

  defp normalize_eval_wdl(%{"win" => win, "draw" => draw, "loss" => loss}),
    do: normalize_wdl([win, draw, loss])

  defp normalize_eval_wdl(%{win: win, draw: draw, loss: loss}),
    do: normalize_wdl([win, draw, loss])

  defp normalize_eval_wdl(_other), do: [0.0, 1.0, 0.0]

  defp cp_to_wdl(cp, game) when is_integer(cp) or is_float(cp) do
    move_phase = max(1.0, min(game.fullmove * 1.0, 58.0))

    a =
      ((-185.71 * move_phase / 58.0 + 504.85) * move_phase / 58.0 - 438.58) * move_phase / 58.0 +
        474.05

    b =
      ((89.24 * move_phase / 58.0 - 137.02) * move_phase / 58.0 + 73.29) * move_phase / 58.0 +
        47.53

    win = logistic((cp - a) / b)
    loss = logistic((-cp - a) / b)
    draw = max(0.0, 1.0 - win - loss)

    normalize_wdl([win, draw, loss]) |> Nx.tensor(type: :f32)
  end

  defp logistic(x), do: 1.0 / (1.0 + :math.exp(-x))

  defp mate_to_q(mate) when is_integer(mate) or is_float(mate) do
    cond do
      mate > 0 -> 1.0
      mate < 0 -> -1.0
      true -> 0.0
    end
  end

  defp result_to_q("1-0", :white), do: 1.0
  defp result_to_q("1-0", :black), do: -1.0
  defp result_to_q("0-1", :white), do: -1.0
  defp result_to_q("0-1", :black), do: 1.0
  defp result_to_q("1/2-1/2", _turn), do: 0.0
  defp result_to_q(_result, _turn), do: 0.0
end
