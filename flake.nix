{
  description = "open-notes: 100% local, offline AI knowledge base with MCP server";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;

        # Check if CUDA is available on this system (simplified check)
        # Default to false - CUDA can be enabled manually if needed
        systemHasCUDA = false;

        # Custom Python environment with all required packages
        # Uses CPU versions by default, CUDA if available
        pythonEnv = pkgs.python3.withPackages (python-packages: with python-packages; [
          # Core dependencies (from pyproject.toml)
          pyyaml
          python-frontmatter
          watchdog
          faiss
          sentence-transformers
          langchain
          langchain-community
          llama-cpp-python
          click
          tqdm

          # Additional dependencies
          numpy
          scipy
          scikit-learn
          torch
          transformers
          huggingface-hub
          jinja2
          diskcache
          typing-extensions

          # sentence-transformers - skip tests to avoid build failures
          (sentence-transformers.overrideAttrs (old: {
            doCheck = false;
          }))

          # llama-cpp-python - use CUDA if available on the system
          (llama-cpp-python.override {
            cudaSupport = systemHasCUDA;
          })

          # MCP SDK
          mcp
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Core tools
            git
            vim
            curl

            # Our Python environment with all dependencies
            pythonEnv
          ] ++ (lib.optionals systemHasCUDA [
            # CUDA toolkit for GPU support (only if available)
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
          ]);

          shellHook = ''
            export PYTHONPATH="$PYTHONPATH:$PWD"
            ${lib.optionalString systemHasCUDA "export CUDA_VISIBLE_DEVICES=0"}
          '';
        };

        packages.default = pkgs.python3.pkgs.buildPythonPackage {
          pname = "open-notes";
          version = "0.1.0";
          format = "pyproject";

          src = ./.;

          dependencies = with pkgs.python3.pkgs; [
            # Core dependencies (from pyproject.toml)
            pyyaml
            python-frontmatter
            watchdog
            faiss
            sentence-transformers
            langchain
            langchain-community
            llama-cpp-python
            click
            tqdm

            # Additional dependencies
            numpy
            scipy
            scikit-learn
            torch
            transformers
            huggingface-hub
            jinja2
            diskcache
            typing-extensions
            mcp
          ];

          meta = with pkgs; {
            description = "100% local, offline, open-source AI knowledge base with MCP server";
            homepage = "https://github.com/Darth-218/open-notes";
            license = lib.licenses.mit;
          };
        };
      }
    );
}