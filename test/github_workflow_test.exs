defmodule EchecsEngine.GitHubWorkflowTest do
  use ExUnit.Case, async: true

  @workflow Path.expand("../.github/workflows/docker-images.yml", __DIR__)

  test "docker image workflow builds cpu, nvidia, and match images on github" do
    workflow = File.read!(@workflow)

    assert workflow =~ "name: Docker Images"
    assert workflow =~ "pull_request:"
    assert workflow =~ "push:"
    assert workflow =~ "concurrency:"
    assert workflow =~ "matrix:"
    assert workflow =~ "- engine"
    assert workflow =~ "- engine-nvidia"
    assert workflow =~ "- engine-match"
    assert workflow =~ "Free disk space for NVIDIA build"
    assert workflow =~ "matrix.service == 'engine-nvidia'"
    assert workflow =~ "docker compose build ${{ matrix.service }}"
    assert workflow =~ "docker/setup-buildx-action"
  end
end
