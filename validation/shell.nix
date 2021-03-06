let
  pkgs = import (fetchTarball
    "https://github.com/NixOS/nixpkgs/archive/df8e3bd110921621d175fad88c9e67909b7cb3d3.tar.gz"
  ) {};

  futhark-rev = "a29f2f580f02bd9ac43201cf9a9eb5c6f1e6347b";
  futhark-src = pkgs.applyPatches {
    name = "futhark-patched";
    src = (fetchTarball "https://github.com/diku-dk/futhark/archive/${futhark-rev}.tar.gz");
    patches = [ ./futhark.patch ];
  };
  futhark-pinned = pkgs.haskellPackages.callPackage futhark-src { suffix = "nightly"; };
in
pkgs.stdenv.mkDerivation {
  name = "shell";
    buildInputs = with pkgs; [
    opencl-headers
    ocl-icd
    futhark-pinned
    gcc
    (python38.withPackages (pypkgs: with pypkgs; [
      numpy
      pyopencl
      statsmodels
    ]))
  ];
} 
