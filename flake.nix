{
  description = "nix declared development environment";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    let
      # Create function to generate the pipenv-included shell with single
      # input: pkgs
      pipenv-wrapped-generate = pkgs:
        let
          inherit (pkgs) lib;
          # The wanted python interpreters are set here. E.g. if you want to
          # add Python 3.7, add 'python37'.
          pythons = with pkgs; [ python38 python39 ];

          # The paths to site-packages are extracted and joined with a colon
          site-packages = lib.concatStringsSep ":"
            (lib.forEach pythons (python: "${python}/${python.sitePackages}"));

          # The paths to interpreters are extracted and joined with a colon
          interpreters = lib.concatStringsSep ":"
            (lib.forEach pythons (python: "${python}/bin"));

          # Create a script with the filename pipenv so that all "pipenv"
          # prefixed commands run the same. E.g. you can use 'pipenv run'
          # normally. The script sets environment variables before passing
          # all arguments to the pipenv executable These environment
          # variables are required for building Python packages with e.g. C
          # -extensions.
        in pkgs.writeScriptBin "pipenv" ''
          CLIB="${pkgs.stdenv.cc.cc.lib}/lib"
          ZLIB="${pkgs.zlib}/lib"
          CERT="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"

          export GIT_SSL_CAINFO=$CERT
          export SSL_CERT_FILE=$CERT
          export CURL_CA_BUNDLE=$CERT
          export LD_LIBRARY_PATH=$CLIB:$ZLIB

          export PYTHONPATH=${site-packages}
          export PATH=${interpreters}:$PATH
          ${pkgs.execline}/bin/exec -a "$0" "${pkgs.pipenv}/bin/pipenv" "$@"
        '';
      # Define the actual development shell that contains the now wrapped
      # pipenv executable 'pipenv-wrapped'
      mkshell = pkgs:
        let
          # Pass pkgs input to pipenv-wrapped-generate function which then
          # returns the pipenv-wrapped package.
          pipenv-wrapped = pipenv-wrapped-generate pkgs;
        in pkgs.mkShell {
          # The development environment can contain any tools from nixpkgs
          # alongside pipenv Here we add e.g. pre-commit and pandoc
          packages = with pkgs; [ pandoc pipenv-wrapped ];

          envrc_contents = ''
            use flake
          '';

          # Define a shellHook that is called every time that development shell
          # is entered. It installs pre-commit hooks and prints a message about
          # how to install python dependencies with pipenv. Lastly, it
          # generates an '.envrc' file for use with 'direnv' which I recommend
          # using for easy usage of the development shell
          shellHook = ''
            ${pkgs.pastel}/bin/pastel paint -n green "
            Run pipenv install to install environment from pipenv.lock
            "
            [[ ! -a .envrc ]] && echo -n "$envrc_contents" > .envrc
          '';
        };
      # Use flake-utils to declare the development shell for each system nix
      # supports e.g. x86_64-linux and x86_64-darwin (but no guarantees are
      # given that it works except for x86_64-linux, which I use).
    in flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages."${system}";
      in {
        devShells.default = mkshell pkgs;
        checks = {
          test-pipenv-wrapped =
            let pipenv-wrapped = pipenv-wrapped-generate pkgs;
            in pkgs.runCommand "test-pipenv-wrapped" { } ''
              ${pipenv-wrapped}/bin/pipenv --help
              ${pipenv-wrapped}/bin/pipenv init -n
              ${pipenv-wrapped}/bin/pipenv check
              mkdir $out
            '';
        };
      });
}
