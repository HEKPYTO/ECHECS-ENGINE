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

    assert workflow =~
             "docker buildx build --platform linux/amd64 --cache-from type=gha,scope=engine"

    assert workflow =~
             "docker buildx build --platform linux/amd64 --cache-from type=gha,scope=engine-nvidia"

    assert workflow =~
             "docker buildx build --platform linux/amd64 --cache-from type=gha,scope=engine-match"

    assert workflow =~ "--cache-to type=gha,mode=max,scope=engine"
    assert workflow =~ "--cache-to type=gha,mode=max,scope=engine-nvidia"
    assert workflow =~ "--cache-to type=gha,mode=max,scope=engine-match"
    assert workflow =~ "Free disk space for large builds"
    assert workflow =~ "actions/checkout@v4"
    assert workflow =~ "docker/setup-buildx-action@v4"
    refute workflow =~ "docker compose build engine-nvidia"
    refute workflow =~ "docker compose build engine"
    refute workflow =~ "docker compose build engine-match"
  end
end
