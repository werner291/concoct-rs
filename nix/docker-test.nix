# NixOS VM test — verify the Docker image works end-to-end.
#
# Boots a VM with Docker, loads the Nix-built image, runs the full
# concoct pipeline on the small test dataset, and checks that output
# clustering is produced and deterministic.
#
#   nix build .#checks.x86_64-linux.docker

{ pkgs, dockerImage, src }:

pkgs.testers.runNixOSTest {
  name = "concoct-docker";

  nodes.machine =
    { ... }:
    {
      virtualisation = {
        docker.enable = true;
        diskSize = 4096;
        memorySize = 2048;
      };
    };

  testScript = ''
    machine.wait_for_unit("docker.service")

    # Load the Nix-built image
    machine.succeed("docker load < ${dockerImage}")
    machine.succeed("docker images | grep concoct-rs")

    # Prepare test data
    machine.succeed("mkdir -p /tmp/testdata /tmp/output")
    machine.succeed("cp ${src}/tests/test_data/coverage /tmp/testdata/coverage")
    machine.succeed("cp ${src}/tests/test_data/composition.fa /tmp/testdata/composition.fa")

    # Smoke test: --help exits cleanly
    machine.succeed(
        "docker run --rm ghcr.io/werner291/concoct-rs:latest --help"
    )

    # Integration test: run concoct on the small dataset
    machine.succeed(
        "docker run --rm "
        "-v /tmp/testdata:/input:ro "
        "-v /tmp/output:/output "
        "ghcr.io/werner291/concoct-rs:latest "
        "--coverage_file /input/coverage "
        "--composition_file /input/composition.fa "
        "--basename /output/ "
        "-c 10 --no_total_coverage --seed 1 --threads 1"
    )

    # Verify clustering output exists
    machine.succeed("test -f /tmp/output/clustering_gt1000.csv")

    # Verify determinism: known hash for this dataset with seed=1
    hash = machine.succeed("sha256sum /tmp/output/clustering_gt1000.csv | cut -d' ' -f1").strip()
    expected = "3bdec94c13d8bba9c0d778381956697f253d178d2338e14ba1e0c07535969058"
    assert hash == expected, f"Output hash mismatch: {hash} != {expected}"
    print(f"Output hash matches: {hash}")
  '';
}
