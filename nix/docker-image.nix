{ dockerTools, testPython, bashInteractive, coreutils, samtools, bedtools, perl }:

dockerTools.buildLayeredImage {
  name = "ghcr.io/werner291/concoct-rs";
  tag = "latest";
  contents = [
    testPython
    bashInteractive
    coreutils
    samtools
    bedtools
    perl
  ];
  config = {
    Entrypoint = [ "concoct" ];
    Labels = {
      "org.opencontainers.image.source" = "https://github.com/werner291/concoct-rs";
    };
  };
}
