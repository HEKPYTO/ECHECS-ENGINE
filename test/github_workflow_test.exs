defmodule EchecsEngine.GitHubWorkflowTest do
  use ExUnit.Case, async: true

  @workflow Path.expand("../.github/workflows/docker-images.yml", __DIR__)

  test "docker image workflow builds cpu, nvidia, and match images on github" do
    workflow = File.read!(@workflow)

    assert workflow =~ "name: Docker Images"
    assert workflow =~ "pull_request:"
    assert workflow =~ "push:"
    assert workflow =~ "concurrency:"
    assert workflow =~ "strategy:"
    assert workflow =~ "matrix:"
    assert workflow =~ "docker compose build engine"
    assert workflow =~ "docker buildx build --platform linux/amd64"
    assert workflow =~ "docker compose build engine-match"
    assert workflow =~ "Free disk space for NVIDIA build"
    assert workflow =~ "actions/checkout@v6"
    assert workflow =~ "docker/setup-buildx-action@v4"
    refute workflow =~ "docker compose build engine-nvidia"
  end
end
