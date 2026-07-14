defmodule EchecsEngine.GitHubWorkflowTest do
  use ExUnit.Case, async: true

  @workflow Path.expand("../.github/workflows/docker-images.yml", __DIR__)
  @rocm_archive_workflow Path.expand("../.github/workflows/rocm-xla-archive.yml", __DIR__)

  test "docker image workflow builds cpu, nvidia, and match images on github" do
    workflow = File.read!(@workflow)

    assert workflow =~ "name: Docker Images"
    assert workflow =~ "pull_request:"
    assert workflow =~ "push:"
    assert workflow =~ "detect-rocm-inputs:"
    assert workflow =~ "dorny/paths-filter@v4"
    assert workflow =~ "rocm_inputs:"
    assert workflow =~ "archive_image:"
    assert workflow =~ "scripts/rocm-xla-image"
    assert workflow =~ "concurrency:"
    assert workflow =~ "strategy:"
    assert workflow =~ "matrix:"

    assert workflow =~
             "docker buildx build --platform linux/amd64 --cache-from type=gha,scope=engine"

    assert workflow =~
             "docker buildx build --platform linux/amd64 --cache-from type=gha,scope=engine-nvidia"

    assert workflow =~
             "docker buildx build --platform linux/amd64 --cache-from type=gha,scope=engine-amd"

    assert workflow =~
             "docker buildx build --platform linux/amd64 --cache-from type=gha,scope=engine-match"

    assert workflow =~ "--cache-to type=gha,mode=max,scope=engine"
    assert workflow =~ "--cache-to type=gha,mode=max,scope=engine-nvidia"
    assert workflow =~ "--cache-to type=gha,mode=max,scope=engine-amd"
    assert workflow =~ "--cache-to type=gha,mode=max,scope=engine-match"
    assert workflow =~ "Free disk space for large builds"
    assert workflow =~ "actions/checkout@v7"
    assert workflow =~ "docker/setup-buildx-action@v4"
    assert workflow =~ "packages: read"
    assert workflow =~ "pull-requests: read"
    assert workflow =~ "Log in to GHCR for the ROCm XLA archive"
    assert workflow =~ "build-amd:"
    assert workflow =~ "needs.detect-rocm-inputs.outputs.rocm_inputs != 'true'"
    refute workflow =~ "--build-arg XLA_BUILD=true"
    refute workflow =~ "docker compose build engine-nvidia"
    refute workflow =~ "docker compose build engine"
    refute workflow =~ "docker compose build engine-match"
  end

  test "ROCm XLA archive workflow publishes a reusable gfx1201 artifact" do
    workflow = File.read!(@rocm_archive_workflow)

    assert workflow =~ "name: ROCm XLA Archive"
    assert workflow =~ "workflow_dispatch:"
    assert workflow =~ "schedule:"
    assert workflow =~ "Dockerfile.rocm"
    assert workflow =~ "mix.lock"
    assert workflow =~ "- scripts/rocm-xla-image"
    assert workflow =~ "packages: write"
    assert workflow =~ "docker/login-action@v3"
    assert workflow =~ "--target xla-archive"
    refute workflow =~ "--build-arg ROCM_AMDGPU_TARGETS=gfx1201"
    assert workflow =~ "docker buildx imagetools inspect"
    assert workflow =~ "Build the AMD application image with the archive"
    assert workflow =~ "Derive the immutable ROCm XLA archive tag"
    assert workflow =~ "--push"
  end
end
